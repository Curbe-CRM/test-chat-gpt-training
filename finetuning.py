import os
import openai
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")


# response_file=openai.File.create(
#   file=open("finetunning.jsonl", "rb"),
#   purpose='fine-tune'
# )
# response = openai.FineTuningJob.create(training_file=response_file.id,
#                                        model="gpt-3.5-turbo",
#                                        suffix='bankgpttest', hyperparameters={'n_epochs':4})


openai.FineTuningJob.delete('ftjob-94AVdYCsgOtexxVqSPBTEuix')
response = openai.FineTuningJob.list_events(id='ftjob-94AVdYCsgOtexxVqSPBTEuix')

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])

# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage

# system_message ="Gala es un chatbot que ayuda a clientes"

# model_name = "ft:gpt-3.5-turbo-0613:personal:bankgpttest:8ANeP6DR"
# chat = ChatOpenAI(model=model_name, temperature=0.1)

# messages = [
#     SystemMessage(content=system_message),
#     HumanMessage(content="que servicios tiene el banco")
# ]

# response = chat(messages)
# print(response.content)


completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal:bankgpttest:8ANeP6DR",
  messages=[
    {"role": "system", "content": "Eres una mujer que atiende clientes de una institucion"},
    {"role": "user", "content": "que servicios tiene austrobank"}
  ]
)
print(completion.choices[0].message)

# import json
# import tiktoken # for token counting
# import numpy as np
# from collections import defaultdict

# data_path = "finetunning.jsonl"

# # Load the dataset
# with open(data_path, 'r', encoding='utf-8') as f:
#     dataset = [json.loads(line) for line in f]

# # Initial dataset stats
# print("Num examples:", len(dataset))
# print("First example:")
# for message in dataset[0]["messages"]:
#     print(message)

# # Format error checks
# format_errors = defaultdict(int)

# for ex in dataset:
#     if not isinstance(ex, dict):
#         format_errors["data_type"] += 1
#         continue
        
#     messages = ex.get("messages", None)
#     if not messages:
#         format_errors["missing_messages_list"] += 1
#         continue
        
#     for message in messages:
#         if "role" not in message or "content" not in message:
#             format_errors["message_missing_key"] += 1
        
#         if any(k not in ("role", "content", "name", "function_call") for k in message):
#             format_errors["message_unrecognized_key"] += 1
        
#         if message.get("role", None) not in ("system", "user", "assistant", "function"):
#             format_errors["unrecognized_role"] += 1
            
#         content = message.get("content", None)
#         function_call = message.get("function_call", None)
        
#         if (not content and not function_call) or not isinstance(content, str):
#             format_errors["missing_content"] += 1
    
#     if not any(message.get("role", None) == "assistant" for message in messages):
#         format_errors["example_missing_assistant_message"] += 1

# if format_errors:
#     print("Found errors:")
#     for k, v in format_errors.items():
#         print(f"{k}: {v}")
# else:
#     print("No errors found")

# encoding = tiktoken.get_encoding("cl100k_base")

# def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
#     num_tokens = 0
#     for message in messages:
#         num_tokens += tokens_per_message
#         for key, value in message.items():
#             num_tokens += len(encoding.encode(value))
#             if key == "name":
#                 num_tokens += tokens_per_name
#     num_tokens += 3
#     return num_tokens

# def num_assistant_tokens_from_messages(messages):
#     num_tokens = 0
#     for message in messages:
#         if message["role"] == "assistant":
#             num_tokens += len(encoding.encode(message["content"]))
#     return num_tokens

# def print_distribution(values, name):
#     print(f"\n#### Distribution of {name}:")
#     print(f"min / max: {min(values)}, {max(values)}")
#     print(f"mean / median: {np.mean(values)}, {np.median(values)}")
#     print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# n_missing_system = 0
# n_missing_user = 0
# n_messages = []
# convo_lens = []
# assistant_message_lens = []

# for ex in dataset:
#     messages = ex["messages"]
#     if not any(message["role"] == "system" for message in messages):
#         n_missing_system += 1
#     if not any(message["role"] == "user" for message in messages):
#         n_missing_user += 1
#     n_messages.append(len(messages))
#     convo_lens.append(num_tokens_from_messages(messages))
#     assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
# print("Num examples missing system message:", n_missing_system)
# print("Num examples missing user message:", n_missing_user)
# print_distribution(n_messages, "num_messages_per_example")
# print_distribution(convo_lens, "num_total_tokens_per_example")
# print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
# n_too_long = sum(l > 4096 for l in convo_lens)
# print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

# MAX_TOKENS_PER_EXAMPLE = 4096

# TARGET_EPOCHS = 4
# MIN_TARGET_EXAMPLES = 100
# MAX_TARGET_EXAMPLES = 25000
# MIN_DEFAULT_EPOCHS = 1
# MAX_DEFAULT_EPOCHS = 25

# n_epochs = TARGET_EPOCHS
# n_train_examples = len(dataset)
# if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
#     n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
# elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
#     n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

# n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
# print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
# print(f"By default, you'll train for {n_epochs} epochs on this dataset")
# print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")