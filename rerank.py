import torch
import torch.distributed as dist
import json
import argparse
import os
import shutil
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.openqa_dataset import get_openqa_dataset, get_one_epoch_dataloader
from utils.dpr_wiki_dataset import get_open_retrieval_wiki_dataset


import random
import numpy
from tqdm import tqdm

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    
class Reranking():
    def __init__(self, args):
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.evidence_dataset = None

        self.args = args
        self.log_interval = args.log_interval
        self.batch_size = 1

        self.load_attributes()
        # self.is_main_builder = dist.get_rank() == 0
        # self.num_total_builders = dist.get_world_size()
        self.temp_dir_name = os.path.join(args.reranker_output_dir, '_tmp_reranker')

    def load_attributes(self):
        print("Loading {} weights".format(self.args.hf_model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_name)

        # In GPT models by EleutherAI, we need to set the pad_token
        # for llama, is 0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = 0

        if self.args.use_fp16:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_name, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        # if self.args.use_gpu:
        #     self.model = self.model.cuda()

        print("Loaded {} weights".format(self.args.hf_model_name))

        self.model.eval() 
        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        
        # self.dataset = load_dataset(self.args.retriever_topk_passages_path)
        self.iteration = self.total_processed = 0

    def track_and_report_progress(self, batch_size):
        """Utility function for tracking progress"""
        self.iteration += 1
        self.total_processed += batch_size
        if self.iteration % self.log_interval == 0:
            print('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed), flush=True)

    def save_shard(self, data, calibrate):
        """
        Save the block data that was created this in this process
        """
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        outpath = os.path.join(self.temp_dir_name, f"reranker_upr_{calibrate}.json")
        with open(outpath, "w") as writer:
            writer.write(json.dumps(data, indent=4) + "\n")

    def do_inference(self, alpha=0.25):
        """Goes through one epoch of the dataloader and adds all data to this instance's BlockData.

        The copy of BlockData is saved as a shard, which when run in a distributed setting will be
        consolidated by the rank 0 process and saved as a final pickled BlockData.
        """
        int_alpha_list = [0.8]
        reranked_answers_list = []
        reranked_answers_list_int = {alpha: [] for alpha in int_alpha_list}
        reranked_answers_conflict_list = []
        reranked_answers_conflict_list_int = {alpha: [] for alpha in int_alpha_list}
        original_answers_list = []
        all_data = []
        positions = []
        
        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            all_ids, all_labels = [], []
            all_conflict_labels = []
            has_answer_list = []
            retriever_scores = []
            max_input_size = -1

            for i, context in enumerate(all_contexts):
                text, title = context.get("text"), context.get("title")
                score = context.get("score")
                retriever_scores.append(float(score))
                # text, title = self.evidence_dataset.id2text[int(context.get("id"))]
                
                passage = "{} {} {}. {}".format(self.args.verbalizer_head, title, text, self.args.verbalizer)
                cids = self.tokenizer(passage,
                                      max_length=512,
                                      truncation=True).input_ids
                # -100 is the negative integer to be ignored when computing cross-entropy loss
                clabel = [-100] * len(cids)

                question = batch['decoder_ids'][0]
                qids = self.tokenizer(question,
                                      max_length=128,
                                      truncation=True).input_ids
                qlabel = qids

                ids = cids + qids
                if self.args.include_eos_token:
                    ids = ids + [self.tokenizer.eos_token_id]
                all_ids.append(ids)

                labels = clabel + qlabel
                conflict_labels = cids + [-100] * len(qids)
                if self.args.include_eos_token:
                    labels = labels + [self.tokenizer.eos_token_id]
                all_labels.append(labels)
                all_conflict_labels.append(conflict_labels)
                
                if len(ids) > max_input_size:
                    max_input_size = len(ids)
                has_answer_list.append(context.get('has_answer'))

            # Pad all_ids and labels
            padded_labels, padded_ids = [], []
            padded_conflict_labels = []
            
            for ids, label, conflict_label in zip(all_ids, all_labels, all_conflict_labels):
                assert len(ids) == len(label)

                if len(label) < max_input_size:
                    label = label + [-100] * (max_input_size - len(label))
                    conflict_label = conflict_label + [-100] * (max_input_size - len(conflict_label))
                    ids = ids + [self.tokenizer.pad_token_id] * (max_input_size - len(ids))

                padded_labels.append(label)
                padded_ids.append(ids)
                padded_conflict_labels.append(conflict_label)

            padded_labels = torch.LongTensor(padded_labels)
            padded_ids = torch.LongTensor(padded_ids)
            padded_conflict_labels = torch.LongTensor(padded_conflict_labels)

            if self.args.use_gpu:
                context_tensor = padded_ids.cuda()
                padded_labels = padded_labels.cuda()
                padded_conflict_labels = padded_conflict_labels.cuda()
            else:
                context_tensor = padded_ids


            sharded_nll_list = []
            sharded_nll_list_int = {alpha: [] for alpha in int_alpha_list}
            sharded_conflict_nll_list = []
            sharded_conflict_nll_list_int = {alpha: [] for alpha in int_alpha_list}
            avg_nll_int = {alpha: [] for alpha in int_alpha_list}
            avg_nll_conflict_int = {alpha: [] for alpha in int_alpha_list}
            # min-max normalization
            min_score = min(retriever_scores)
            max_score = max(retriever_scores)
            scores_normalized = [(score - min_score) / (max_score - min_score) for score in retriever_scores]
            
            for i in range(0, len(context_tensor), self.args.shard_size):
                encoder_tensor_view = context_tensor[i: i + self.args.shard_size]
                labels_view = padded_labels[i: i + self.args.shard_size]
                conflict_labels_view = padded_conflict_labels[i: i + self.args.shard_size]
                # scores_view = torch.FloatTensor(scores_normalized[i: i + self.args.shard_size]).cuda()
                
                with torch.no_grad():
                    logits = self.model(input_ids=encoder_tensor_view).logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels_view[..., 1:].contiguous()
                    shift_conflict_labels = conflict_labels_view[..., 1:].contiguous()
                    

                    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                    nll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nll = nll.view(shift_labels.size())
                    nll_with_nan = nll.clone()
                    nll_with_nan[nll_with_nan == 0] = float('nan')
                    avg_nll = torch.nanmean(nll_with_nan, dim=1)
                    
                    
                    
                    
                    #conflict
                    nll_conflict = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_conflict_labels.view(-1))
                    nll_conflict = nll_conflict.view(shift_conflict_labels.size())
                    nll_conflict_with_nan = nll_conflict.clone()
                    # nll_conflict_with_nan[nll_conflict_with_nan == -float('inf')] = float('nan')
                    nll_conflict_with_nan[nll_conflict_with_nan == 0] = float('nan')
                    # avg_nll_conflict = torch.nanmean(nll_conflict_with_nan, dim=1) + avg_nll
                    avg_nll_conflict = torch.nanmean(nll_conflict_with_nan, dim=1) * alpha + avg_nll
                
                sharded_nll_list.append(avg_nll)
                
                sharded_conflict_nll_list.append(avg_nll_conflict)
                # for int_alpha in int_alpha_list:
                #     sharded_conflict_nll_list_int[int_alpha].append(avg_nll_conflict_int[int_alpha])
            avg_nll_list = -torch.cat(sharded_nll_list)
            avg_nll_conflict_list = -torch.cat(sharded_conflict_nll_list)
            # min-max normalization
            min_avg_nll = torch.min(avg_nll_list)
            max_avg_nll = torch.max(avg_nll_list)
            avg_nll_normalized = (avg_nll_list - min_avg_nll) / (max_avg_nll - min_avg_nll)
            
            min_avg_nll_conflict = torch.min(avg_nll_conflict_list)
            max_avg_nll_conflict = torch.max(avg_nll_conflict_list)
            avg_nll_conflict_normalized = (avg_nll_conflict_list - min_avg_nll_conflict) / (max_avg_nll_conflict - min_avg_nll_conflict)
            # inter
            scores_view = torch.FloatTensor(scores_normalized).cuda()
            
            for int_alpha in int_alpha_list:
                sharded_nll_list_int[int_alpha] = avg_nll_normalized * int_alpha + scores_view * (1 - int_alpha)
                sharded_conflict_nll_list_int[int_alpha] = avg_nll_conflict_normalized * int_alpha + scores_view * (1 - int_alpha)
            
            topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))
            for int_alpha in int_alpha_list:
                topk_scores_int, indexes_int = torch.topk(sharded_nll_list_int[int_alpha], k=len(context_tensor))
                ranked_answers_int = torch.BoolTensor(has_answer_list)[indexes_int.cpu()]
                reranked_answers_list_int[int_alpha].append(ranked_answers_int.tolist())
                
                _, indexes_conflict_int = torch.topk(sharded_conflict_nll_list_int[int_alpha], k=len(context_tensor)) 
                ranked_answers_conflict_int = torch.BoolTensor(has_answer_list)[indexes_conflict_int.cpu()]
                reranked_answers_conflict_list_int[int_alpha].append(ranked_answers_conflict_int.tolist())                
                
            topk_scores_conflict, indexes_conflict = torch.topk(-torch.cat(sharded_conflict_nll_list), k=len(context_tensor))
            ranked_answers = torch.BoolTensor(has_answer_list)[indexes.cpu()]
            ranked_answers_conflict = torch.BoolTensor(has_answer_list)[indexes_conflict.cpu()]
            

            # Save the essential information to be used for saving the re-ranked information component.
            original_answers_list.append(has_answer_list)
            reranked_answers_list.append(ranked_answers.tolist())
            
            reranked_answers_conflict_list.append(ranked_answers_conflict.tolist())
            
            

            self.track_and_report_progress(batch_size=len(batch['id']))
        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")
        for alpha in int_alpha_list:
            print(alpha)
            self.compute_topk_recall(reranked_answers_list_int[alpha], string_prefix="Re-Ranking-int")
        self.compute_topk_recall(reranked_answers_conflict_list, string_prefix="Re-Ranking by conflict")
        for alpha in int_alpha_list:
            print(alpha)
            self.compute_topk_recall(reranked_answers_conflict_list_int[alpha], string_prefix="Re-Ranking by conflict-int")

        self.compute_topk_ndcg(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_ndcg(reranked_answers_list, string_prefix="Re-Ranking")
        for alpha in int_alpha_list:
            print(alpha)
            self.compute_topk_ndcg(reranked_answers_list_int[alpha], string_prefix="Re-Ranking-int")
        self.compute_topk_ndcg(reranked_answers_conflict_list, string_prefix="Re-Ranking by conflict")
        for alpha in int_alpha_list:
            print(alpha)
            self.compute_topk_ndcg(reranked_answers_conflict_list_int[alpha], string_prefix="Re-Ranking by conflict-int")
        
        self.compute_map(original_answers_list, string_prefix="Original Ranking")
        self.compute_map(reranked_answers_list, string_prefix="Re-Ranking")
        for alpha in int_alpha_list:
            print(alpha)
            self.compute_map(reranked_answers_list_int[alpha], string_prefix="Re-Ranking-int")
        self.compute_map(reranked_answers_conflict_list, string_prefix="Re-Ranking by conflict")
        for alpha in int_alpha_list:
            print(alpha)
            self.compute_map(reranked_answers_conflict_list_int[alpha], string_prefix="Re-Ranking by conflict-int")
        print("model: ", self.args.hf_model_name)
        print("data: ", self.args.retriever_topk_passages_path)
        print("hyperparamter: ", int_alpha_list)
        del self.model
    
    
    @staticmethod
    def dcg_at_k(scores, k, device='cuda'):
        ranks = torch.arange(2, k + 2, device=device, dtype=torch.float) 
        discounts = torch.log2(ranks)
        return (scores[:, :k] / discounts).sum(dim=1)

    def ndcg_at_k(self, scores, k, device='cuda'):
        sorted_scores, _ = torch.sort(scores, descending=True, dim=1)
        ideal_dcg = self.dcg_at_k(sorted_scores, k, device)
        actual_dcg = self.dcg_at_k(scores, k, device)
        ndcg = actual_dcg / ideal_dcg
        ndcg[torch.isnan(ndcg)] = 0 
        return ndcg.mean().item()  


    def compute_topk_ndcg(self, scores_list, string_prefix):
        max_k=self.args.report_topk_accuracies[-1]
        # topk_scores = self.calculate_scores(scores_list, max_k=max_k)
        topk_scores = torch.FloatTensor(scores_list).cuda()
        print(string_prefix)
        for i in self.args.report_topk_accuracies:
            ndcg_value = self.ndcg_at_k(topk_scores, i)
            print("NDCG@{}: {:.2f}".format(i, ndcg_value * 100))
        print("\n")

    @staticmethod
    def calculate_average_precision(scores):
        relevant = scores > 0  
        # order = torch.argsort(scores, descending=True)
        # sorted_relevant = relevant[order]

        precisions = torch.cumsum(relevant.float(), dim=0) / torch.arange(1, scores.size(0) + 1).float().cuda()
        if relevant.sum() == 0:
            return 0.0  
        return (precisions * relevant.float()).sum() / relevant.sum()

    def compute_map(self, scores_list, string_prefix):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scores_tensor = torch.FloatTensor(scores_list).to(device)
        ap_values = [self.calculate_average_precision(scores_tensor[i]) for i in range(scores_tensor.size(0))]
        map_value = sum(ap_values) / len(ap_values) if len(ap_values) > 0 else 0
        print(f"{string_prefix} MAP: {map_value:.4f}\n")


    @staticmethod
    def calculate_topk_hits(scores, max_k):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits

    def compute_topk_recall(self, answers_list, string_prefix):
        topk_hits = self.calculate_topk_hits(answers_list, max_k=self.args.report_topk_accuracies[-1])
        topk_hits = torch.FloatTensor(topk_hits).cuda()
        # torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)

        n_docs = torch.FloatTensor([len(answers_list)]).cuda()
        # torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

        # if torch.distributed.get_rank() == 0:
        topk_hits = topk_hits / n_docs

        print(string_prefix)
        for i in self.args.report_topk_accuracies:
            print("top-{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
        print("\n")

    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        all_data = []

        for fname in os.listdir(self.temp_dir_name):
            shard_size = 0
            old_size = len(all_data)
            fpath = '{}/{}'.format(self.temp_dir_name, fname)
            with open(fpath, 'r') as f:
                data = json.load(f)
                shard_size = len(data)
                all_data.extend(data)

            assert len(all_data) == old_size + shard_size
            os.remove(fpath)

        # save the consolidated shards
        outpath = os.path.join(self.args.output_path, "{}.json".format(self.args.special_suffix))

        with open(outpath, 'w') as writer:
            writer.write(json.dumps(all_data, indent=4) + "\n")

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(all_data)), flush=True)

        # make sure that every single piece of data was embedded
        assert len(all_data) == len(self.dataset)

        
def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='output data')

    group.add_argument('--local_rank', type=int, default=0,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--main-port', type=int, default=29500,
                       help='Main port number.')

    group.add_argument('--special-suffix', type=str, default="",
                       help='special suffix extension for saving merged file')

    group.add_argument('--retriever-topk-passages-path', type=str, default="./",
                       help='Path of the Top-K outputs from retriever (.json file)')

    group.add_argument('--topk-passages', type=int, default=100,
                       help='number of topk context to select')

    group.add_argument('--log-interval', type=int, default=10,
                       help='Interval between progress updates')

    group.add_argument('--shard-size', type=int, default=20)

    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    group.add_argument('--reranker-output-dir', type=str, default="downloads/data/retriever-outputs/",
                       help='Where to save inference results')

    group.add_argument('--task-name', type=str, default="reranking",
                       help='Name of the task.')

    group.add_argument('--hf-model-name', type=str, default="./llama-2-7b-hf",
                       help='Name of the HF model.')

    group.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU or not')

    group.add_argument('--interactive-node', action='store_true',
                       help='If the node is interactive or not')

    group.add_argument('--use-fp16', action='store_true', default=True,
                       help='Use FP16 or not')

    group.add_argument('--merge-shards-and-save', action='store_true',
                       help='whether to merge individual data shards or not for reranking')

    group.add_argument('--sample-rate', type=float, default=1.,
                       help="Sample rate for the number of examples.")

    group.add_argument('--random-seed', type=int, default=1234,
                       help="Random seed.")

    group.add_argument('--verbalizer', type=str, default="Please write a question based on this passage. Question:",
                       help='Prompt string for generating the target tokens')

    group.add_argument('--verbalizer-head', type=str, default="Passage: ",
                       help='The string token used to represent encoder input')

    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 2, 3, 4, 5, 10, 20, 50, 100],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    group.add_argument('--include-eos-token', action='store_true',
                       help='whether to include EOS token when calculating question generation likelihood')

    args = parser.parse_args()
    args.keep_empty = False

    return args


def main():
    """
    """
    args = get_args()
    set_random_seed(args.random_seed)
    reranker = Reranking(args)
    reranker.do_inference()


if __name__ == "__main__":
    main()