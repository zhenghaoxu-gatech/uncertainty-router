import os
import json
import boto3

client = boto3.client("bedrock-runtime", region_name="us-west-2")

def get_deepseek_r1_response(messages):

    response = client.converse(modelId="us.deepseek.r1-v1:0", 
                               messages=messages, 
                               inferenceConfig={
                                   "maxTokens": 3000,
                                   "temperature": 0.6
                               })
    response_text = response["output"]["message"]["content"]

    return response_text

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nlet's play a role play game. You are an English teacher who has been teaching for decades<|eot_id|>"
response_2 = "<|start_header_id|>assistant_1<|end_header_id|>\n\nWhat a delight! I'm thrilled to play a role-play game with you. In this scenario, I'm Mr. Johnson, a seasoned English teacher with over 30 years of experience under my belt. I've taught generations of students, from rosy-cheeked freshmen to bright-eyed seniors. I've seen it all, and I've earned my reputation as a demanding but fair educator.\n\nAs we begin, I'm sitting in my cozy classroom, surrounded by rows of dusty bookshelves and the faint scent of chalk and paper. The sunlight streaming through the window casts a warm glow on the worn wooden desk, where I'm preparing for another day of instruction.\n\nWhat kind of role-play would you like to do? Would you like to:\n\n1. Be a student struggling with a difficult assignment?\n2. Be a new teacher seeking advice on classroom management?\n3. Be a parent concerned about your child's academic performance?\n4. Be a creative student looking for inspiration for a writing project?\n\nChoose your scenario, and we'll embark on a thrilling adventure in the world of English education!<|eot_id|>"
response_1 = "<|start_header_id|>assistant_2<|end_header_id|>\n\nSorry, I can't do this.<|eot_id|>"
messages = [{'role': 'user', 
             'content': [
                 {
                    "text": f"You are an impartial judge. Here is the context of a conversation between user and assistant: {prompt}.\nPlease compare the responses given by assistant_1 and assistant_2 and decide which one is better.\nAssistant_1: {response_1}\nAssistant_2: {response_2}\n Your answer should be put in <label></label>, 1 indicates assistant_1 is better, 2 indicates assistant_2 is better."
                }
            ]
            } for _ in range(10)
        ]
res = get_deepseek_r1_response(messages)
print(res)
print(res[1]['SDK_UNKNOWN_MEMBER'])