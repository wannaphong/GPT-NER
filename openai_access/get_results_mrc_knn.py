import os
from tqdm import tqdm
from base_access import AccessBase
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random

random.seed(1)
logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-dir", type=str, help="directory for the input")
    parser.add_argument("--source-name", type=str, help="file name for the input")
    parser.add_argument("--train-name", type=str, default="None", help="file name for the training set")
    parser.add_argument("--data-name", type=str, help="dataset name for the input")
    parser.add_argument("--example-dir", type=str, default="None", help="directory for the example")
    parser.add_argument("--example-name", type=str, default="None", help="file name for the example")
    parser.add_argument("--example-num", type=int, default=16, help="numebr for examples")
    parser.add_argument("--last-results", type=str, default="None", help="unfinished file")
    parser.add_argument("--write-dir", type=str, help="directory for the output")
    parser.add_argument("--write-name", type=str, help="file name for the output")
    
    # Batch processing arguments
    parser.add_argument("--use-batch", action="store_true", help="use OpenAI Batch API for processing")
    parser.add_argument("--batch-file", type=str, default=None, help="path to save/load batch JSONL file")
    parser.add_argument("--batch-id", type=str, default=None, help="existing batch ID to retrieve results from")
    parser.add_argument("--wait-for-batch", action="store_true", help="wait for batch completion (use with --use-batch)")
    
    return parser

def read_mrc_data(dir_, prefix="test"):
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_results(dir_):
    file = open(dir_, "r")
    resulst = file.readlines()
    file.close()
    return resulst

def read_examples(dir_, prefix="dev"):
    print("reading ...")
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

def read_idx(dir_, prefix="test"):
    print("reading ...")
    file_name = os.path.join(dir_, f"{prefix}.knn.jsonl")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx

def mrc2prompt(mrc_data, data_name="CONLL", example_idx=None, train_mrc_data=None, example_num=16, last_results=None):
    print("mrc2prompt ...")

    def get_example(index):
        exampel_prompt = ""
        for idx_ in example_idx[index][:example_num]:
            context = train_mrc_data[idx_]["context"]
            context_list = context.strip().split()
            labels = ""

            last_ = 0
            for span_idx in range(len(train_mrc_data[idx_]["start_position"])):
                start_ = train_mrc_data[idx_]["start_position"][span_idx]
                end_ = train_mrc_data[idx_]["end_position"][span_idx] + 1
                if labels != "":
                    labels += " "
                if last_ == start_:
                    labels += "@@" + " ".join(context_list[start_:end_]) + "##"
                else:
                    labels += " ".join(context_list[last_:start_]) + " @@" + " ".join(context_list[start_:end_]) + "##"
                last_ = end_

            if labels != "" and last_ != len(context_list):
                labels += " "
            labels += " ".join(context_list[last_:])

            exampel_prompt += f"The given sentence: {context}\n"
            
            # exampel_prompt += f"{prompt_label_name} entities: {labels}\n"
            exampel_prompt += f"The labeled sentence: {labels}\n"
        return exampel_prompt
        
    results = []
    for item_idx in tqdm(range(len(mrc_data))):

        if last_results is not None and last_results[item_idx].strip() != "FRIDAY-ERROR-ErrorType.unknown":
            continue

        item_ = mrc_data[item_idx]
        context = item_["context"]
        origin_label = item_["entity_label"]
        transfered_label, sub_prompt = FULL_DATA[data_name][origin_label]
        prompt_label_name = transfered_label[0].upper() + transfered_label[1:]
        # prompt = f"I want to extract {transfered_label} entities that {sub_prompt}, and if that does not exist output \"none\". Below are some examples.\n"
        # prompt = f"I want to extract {transfered_label} entities that {sub_prompt}. Below are some examples.\n"
        prompt = f"You are an excellent linguist. Within the OntoNotes5.0 dataset, the task is to label {transfered_label} entities that {sub_prompt}. Below are some examples, and you should make the same prediction as the examples.\n"
        # prompt = f"You are an excellent linguist. The task is to label {transfered_label} entities in the given sentence. {prompt_label_name} entities {sub_prompt}. Noted that if the given sentence does not contain any {transfered_label} entities, just output the same sentence, or surround the extracted entities by @@ and ## if there exist {transfered_label} entities. Below are some examples."
        # prompt = f"You are an excellent linguistic. The task is to label {transfered_label} entities that {sub_prompt}. First, articulate the clues and reasoning process for determining {transfered_label} entities in the sentence. Next, based on the clues and your reasoning process, label {transfered_label} entities in the sentence. Below are some examples.\n"

        # prompt += get_knn(test_sentence=context, nums=example_false, label_name=transfered_label, positive_idx=0)
        # prompt += get_knn(test_sentence=context, nums=example_true, label_name=transfered_label, positive_idx=1)
        prompt += get_example(index=item_idx)

        # context_list = context.strip().split()
        # index_string = ""
        # for sub_idx in range(len(context_list)):
        #     index_string += f"{context_list[sub_idx]} {sub_idx}\n"
        # prompt += f"The given sentence: {context}\n{index_string}{prompt_label_name} entities:"
        prompt += f"The given sentence: {context}\nThe labeled sentence:"
        # prompt += f"The given sentence: {context}\nThe labeled sentence:"

        # print(prompt)
        results.append(prompt)
    
    return results

def ner_access(openai_access, ner_pairs, use_batch=False, batch_file=None, batch_id=None, wait_for_batch=False):
    """
    Process NER prompts using either parallel requests or batch processing.
    
    Args:
        openai_access: AccessBase instance
        ner_pairs: List of prompts to process
        use_batch: If True, use Batch API instead of parallel processing
        batch_file: Path to save/load batch JSONL file (required if use_batch=True)
        batch_id: Existing batch ID to retrieve results from (optional)
        wait_for_batch: If True, wait for batch completion before returning (use with use_batch=True)
    
    Returns:
        List of results, or batch_id string (if use_batch=True and not waiting)
    """
    print("tagging ...")
    
    if batch_id:
        # Retrieve results from an existing batch
        print(f"Retrieving results from batch {batch_id}...")
        return openai_access.retrieve_batch_results(batch_id)
    
    if use_batch:
        # Use Batch API for processing
        if not batch_file:
            raise ValueError("--batch-file is required when using --use-batch")
        
        print(f"Creating batch file: {batch_file}")
        openai_access.create_batch_file(ner_pairs, batch_file)
        
        print(f"Submitting batch...")
        batch_id = openai_access.submit_batch(batch_file, description="GPT-NER MRC KNN processing")
        print(f"Batch submitted with ID: {batch_id}")
        
        if wait_for_batch:
            print("Waiting for batch completion...")
            final_status = openai_access.wait_for_batch(batch_id)
            print(f"Batch completed with status: {final_status['status']}")
            
            if final_status['status'] == 'completed':
                print("Retrieving results...")
                return openai_access.retrieve_batch_results(batch_id)
            else:
                raise RuntimeError(f"Batch processing failed with status: {final_status['status']}")
        else:
            # Return batch ID for later retrieval
            print(f"\nBatch is processing. To retrieve results later, run:")
            print(f"  python get_results_mrc_knn.py --batch-id {batch_id} --write-dir <dir> --write-name <name>")
            return batch_id
    else:
        # Use parallel processing (default)
        return openai_access.get_multiple_sample(ner_pairs)

def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(line.strip()+'\n')
    file.close()
    # json.dump(labels, open(file_name, "w"), ensure_ascii=False)

def test():
    openai_access = AccessBase(
        engine="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        max_workers=10
    )

    ner_test = read_mrc_data("/data2/wangshuhe/gpt3_ner/gpt3-data/conll_mrc/low_resource", prefix="test")[:4]
    mrc_train = read_mrc_data("/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en", prefix="train.8")
    example_idx = read_idx("/home/wangshuhe/gpt-ner/openai_access/low_resource_data/conll_en", prefix="test.8.embedding")

    prompts = mrc2prompt(mrc_data=ner_test, data_name="CONLL", example_idx=example_idx, train_mrc_data=mrc_train, example_num=4)
    results = ner_access(openai_access=openai_access, ner_pairs=prompts)
    print(results)

if __name__ == '__main__':
    # test()

    parser = get_parser()
    args = parser.parse_args()

    openai_access = AccessBase(
        engine="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        max_workers=10
    )

    # Handle batch retrieval mode
    if args.batch_id:
        print(f"Retrieving results from batch {args.batch_id}...")
        results = ner_access(openai_access, [], batch_id=args.batch_id)
        write_file(results, args.write_dir, args.write_name)
        print(f"Results written to {args.write_dir}/{args.write_name}")
    else:
        # Normal processing mode (parallel or batch)
        ner_test = read_mrc_data(args.source_dir, prefix=args.source_name)
        mrc_train = read_mrc_data(dir_=args.source_dir, prefix=args.train_name)
        example_idx = read_idx(args.example_dir, args.example_name)

        last_results = None
        if args.last_results != "None":
            last_results = read_results(dir_=args.last_results)

        prompts = mrc2prompt(mrc_data=ner_test, data_name=args.data_name, example_idx=example_idx, train_mrc_data=mrc_train, example_num=args.example_num, last_results=last_results)
        
        results = ner_access(
            openai_access=openai_access, 
            ner_pairs=prompts,
            use_batch=args.use_batch,
            batch_file=args.batch_file,
            wait_for_batch=args.wait_for_batch
        )
        
        # Handle different return types
        if isinstance(results, str):
            # Batch ID returned (batch submitted without wait)
            print(f"\nBatch submitted with ID: {results}")
            print(f"Results will be available after batch completes.")
        else:
            # Results list returned (parallel mode or batch with wait)
            write_file(results, args.write_dir, args.write_name)
            print(f"Results written to {args.write_dir}/{args.write_name}")