from datasets import load_dataset, Dataset, DatasetDict
import json
import matplotlib.pyplot as plt
from collections import Counter
import os

def create_message_format(prompt, response):
    """Convert prompt and response to message format"""
    return [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": response
        }
    ]

def create_generative_format(prompt, response_1, response_2, overall_preference):
    """Create the generative format entry"""
    # Combine prompt and responses into a single question
    combined_question = f"""You are an impartial judge. Your task is to compare the quality of two responses to a prompt, provided within tags <response_1></response_1>, <response_2></response_2>, and <prompt></prompt>, respectively. You should give a label from -3 to 3, where:
-3: You are confident that Response 1 is much better than Response 2;
-2: You are confident that Response 1 is better than Response 2;
-1: You are confident that Response 1 is slightly better than Response 2;
0: You are confident that Response 1 and Response 2 are of similar quality, or you cannot decide which one is better due to lack of knowledge;
1: You are confident that Response 2 is slightly better than Response 1;
2: You are confident that Response 2 is better than Response 1;
3: You are confident that Response 2 is much better than Response 1;

Please think step by step and then provide your reasoning process and label strictly following the format:<think>your reasoning process here</think><label>your label here</label>.
Here is the helpsteer3 rubric that can help you think: 

"Ratings should be made by prioritizing the response characteristics below in the order they are stated. Trade-offs should favor higher priority characteristics.
1. Instruction Following
- Responses should follow all requests in the prompt, following the instructions in the prompt is the primary criteria for ranking responses.
- Many prompts will have a clear instruction that has additional or implicit requirements, e.g. "plan an itinerary for a 5-day trip to Portugal that focuses on surfing" should be evaluated on whether an itinerary is produced, whether it is 5 days long and in Portugal, and also whether it includes a lot of surfing - All instructions should be considered when ranking responses, with the core instructions being weighed more heavily.
- If specific formatting (table, json) is asked for in a prompt that should be considered as the high priority "instruction following" and not general "formatting" which is lower priority.
2. Correctness
- Answers which are factually correct should be rated higher than answers that are incorrect.
- Annotators should search the internet when they are unsure about correctness.
- Misleading or incomplete answers are considered less correct than complete answers.
- If prompt (question) contains false premise, the response that pushes back on it should be preferred.
- When a question cannot be answered definitively, the response which expresses uncertainty should be preferred.
3. Formatting
- When no specific formatting is requested responses with better formatting are preferred.
- Vendor tool should render markdown for easier assessment of formatting (markdown tables, lists, shell scripts, etc. should be rendered properly).
- Formatting encompasses anything that is appropriate for the response, e.g. an itinerary being split by day, including a markdown table when it makes sense but was not asked for, any appropriate use of markdown, responding with a shell script, etc.
4. Clarity
- Answers that are easier to understand, provide more clarity (via explanation or introduction), or are generally better written are preferred.
- Unnecessary verbosity, or providing extra information that is irrelevant should be penalized."

<prompt>{prompt}</prompt>
<response_1>{response_1}</response_1>
<response_2>{response_2}</response_2>"""
    
    context_messages = [
        # {
        #     "role": "system",
        #     "content": "Please reason step by step, and put your final answer within \\boxed{}."
        # },
        {
            "role": "user",
            "content": combined_question
        }
    ]
    
    return {
        "context": prompt,
        "context_messages": context_messages,
        "label": str(overall_preference)
    }


def plot_strength_histogram(strengths, split_name):
    """Create and save histogram of preference strengths"""
    # Count frequencies
    strength_counts = Counter(strengths)
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(strength_counts.keys(), strength_counts.values())
    
    # Customize the plot
    plt.title(f'Distribution of Preference Strengths ({split_name} split)')
    plt.xlabel('Preference Strength')
    plt.ylabel('Count')
    
    # Add count labels on top of each bar
    for i, v in strength_counts.items():
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    # Create stats directory if it doesn't exist
    os.makedirs('downloads/stats', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'downloads/stats/strength_histogram_{split_name}.png')
    plt.close()
    
    # Save the counts as JSON
    with open(f'downloads/stats/strength_counts_{split_name}.json', 'w') as f:
        json.dump(dict(strength_counts), f, indent=2)
    
    return dict(strength_counts)

def process_dataset(dataset_items, split_name):
    # First format (chosen, rejected, strength)
    format1_data = []
    # Second format (prompt, response_1, response_2, strength, label)
    format2_data = []
    # Third format (context, context_messages, label)
    format3_data = []
    
    # List to store preference strengths for statistics
    strengths = []
    
    for item in dataset_items:
        strengths.append(item['overall_preference'])
        if item['overall_preference'] == 0:
            continue
        
        # # For the first format (preference data)
        # if item['overall_preference'] > 0:
        #     chosen = create_message_format(item['prompt'], item['response2'])
        #     rejected = create_message_format(item['prompt'], item['response1'])
        #     strength = abs(item['overall_preference'])
        #     format1_entry = {
        #         "chosen": chosen,
        #         "rejected": rejected,
        #         "strength": strength
        #     }
        #     format1_data.append(format1_entry)
        # elif item['overall_preference'] < 0:
        #     chosen = create_message_format(item['prompt'], item['response1'])
        #     rejected = create_message_format(item['prompt'], item['response2'])
        #     strength = abs(item['overall_preference'])
        #     format1_entry = {
        #         "chosen": chosen,
        #         "rejected": rejected,
        #         "strength": strength
        #     }
        #     format1_data.append(format1_entry)
        
        # For the second format (comparison data)
        # Original order
        text_1 = item['context'] + [
            {
                "role": "assistant_1",
                "content": item['response1']
            },
            {
                "role": "assistant_2",
                "content": item['response2']
            },
        ]

        format2_entry1 = {
            "context": item['context'][0]['content'],
            "context_messages": text_1,
            "strength": max(abs(item['overall_preference']), 1),
            "label": 0.5 if item['overall_preference'] == 0 else (0. if item['overall_preference'] > 0 else 1.)
        }
        format2_data.append(format2_entry1)
        
        # Reversed order
        text_2 = item['context'] + [
            {
                "role": "assistant_1",
                "content": item['response2']
            },
            {
                "role": "assistant_2",
                "content": item['response1']
            },
        ]
        format2_entry2 = {
            "context": item['context'][0]['content'],
            "context_messages": text_2,
            "strength": max(abs(item['overall_preference']), 1),
            "label": 0.5 if item['overall_preference'] == 0 else (1. if item['overall_preference'] > 0 else 0.)
        }
        format2_data.append(format2_entry2)
        
        # # For the third format (generative data)
        # # Original order
        # format3_entry1 = create_generative_format(
        #     item['prompt'], 
        #     item['response1'], 
        #     item['response2'], 
        #     item['overall_preference']
        # )
        # format3_data.append(format3_entry1)
        
        # # Reversed order
        # format3_entry2 = create_generative_format(
        #     item['prompt'], 
        #     item['response2'], 
        #     item['response1'], 
        #     -item['overall_preference']  # Flip the sign for reversed order
        # )
        # format3_data.append(format3_entry2)
    
    # Generate and save statistics
    strength_counts = plot_strength_histogram(strengths, split_name)
    
    # Print statistics
    print(f"\nStatistics for {split_name} split:")
    print(f"Total samples: {len(strengths)}")
    print("Strength distribution:")
    for strength, count in strength_counts.items():
        print(f"Strength {strength}: {count} samples ({count/len(strengths)*100:.2f}%)")
    print(f"Number of preference pairs (excluding ties): {len(format1_data)}")
    print(f"Number of comparison pairs (including ties): {len(format2_data)}")
    
    return format1_data, format2_data, format3_data


def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_and_push_dataset(format1_data_dict, format2_data_dict, format3_data_dict, repo_name):
    """Create and push datasets to Hugging Face Hub"""
    # Create Dataset objects for all formats
    # preference_dataset = DatasetDict({
    #     'train': Dataset.from_list(format1_data_dict['train']),
    #     'validation': Dataset.from_list(format1_data_dict['validation'])
    # })
    
    comparison_dataset = DatasetDict({
        'train': Dataset.from_list(format2_data_dict['train']),
        'validation': Dataset.from_list(format2_data_dict['validation'])
    })
    
    # generative_dataset = DatasetDict({
    #     'train': Dataset.from_list(format3_data_dict['train']),
    #     'validation': Dataset.from_list(format3_data_dict['validation'])
    # })
    
    # Push to Hub
    # preference_dataset.push_to_hub(f"{repo_name}_preference")
    comparison_dataset.push_to_hub(f"{repo_name}_comparison")
    # generative_dataset.push_to_hub(f"{repo_name}_generative2")
    
    print(f"Datasets pushed to Hugging Face Hub under {repo_name}_preference, {repo_name}_comparison, and {repo_name}_generative")

def main():
    # Create output directories
    os.makedirs('downloads/data', exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("nvidia/HelpSteer3")
    
    train_items = dataset['train']
    valid_items = dataset['validation']
    # Filter dataset based on split column
    # train_items = [item for item in dataset if item['split'] == 'train']
    # valid_items = [item for item in dataset if item['split'] == 'val']
    
    # Process each split
    format1_data_dict = {}
    format2_data_dict = {}
    format3_data_dict = {}
    
    # Process train split
    format1_data_dict['train'], format2_data_dict['train'], format3_data_dict['train'] = process_dataset(train_items, 'train')
    
    # Process validation split
    format1_data_dict['validation'], format2_data_dict['validation'], format3_data_dict['validation'] = process_dataset(valid_items, 'validation')
    
    # Save separate split files
    for split in ['train', 'validation']:
        save_to_jsonl(format1_data_dict[split], 
                     f'downloads/data/helpsteer3_preference_{split}.jsonl')
        save_to_jsonl(format2_data_dict[split], 
                     f'downloads/data/helpsteer3_comparison_{split}.jsonl')
        save_to_jsonl(format3_data_dict[split], 
                     f'downloads/data/helpsteer3_generative_{split}.jsonl')
        print(f"Processed and saved {split} split")
    
    # Save merged files
    all_format1_data = format1_data_dict['train'] + format1_data_dict['validation']
    all_format2_data = format2_data_dict['train'] + format2_data_dict['validation']
    all_format3_data = format3_data_dict['train'] + format3_data_dict['validation']
    
    save_to_jsonl(all_format1_data, 'downloads/data/helpsteer3_preference_all.jsonl')
    save_to_jsonl(all_format2_data, 'downloads/data/helpsteer3_comparison_all.jsonl')
    save_to_jsonl(all_format3_data, 'downloads/data/helpsteer3_generative_all.jsonl')
    print("Saved merged files")
    
    # Upload to Hugging Face Hub
    create_and_push_dataset(format1_data_dict, format2_data_dict, format3_data_dict, "zhenghaoxu/helpsteer3-preference")

if __name__ == "__main__":
    main()