import json
import pandas as pd
from datasets import Dataset, load_dataset

# Source: https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2maxlen.json
dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

# Templates based on https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json
context_prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要",
    "lcc": "Please complete the code given below. \n{context}",
    "repobench-p": "Please complete the code given below. \n{context}"
}

question_prompt = {
    "narrativeqa": "\n\nQuestion: {input}\n\n",
    "qasper": "\n\nQuestion: {input}\n\n",
    "multifieldqa_en": "\n\nQuestion: {input}\n",
    "multifieldqa_zh": "\n\n问题：{input}\n",
    "hotpotqa": "\n\nQuestion: {input}\n",
    "2wikimqa": "\n\nQuestion: {input}\n",
    "musique": "\n\nQuestion: {input}\n",
    "dureader": "\n\n问题：{input}\n",
    "gov_report": "\n\n{input}", # fake placeholder
    "qmsum": "\n\nQuery: {input}\n",
    "multi_news": "\n\n{input}", # fake placeholder
    "vcsum": "\n\n{input}", # fake placeholder
    "trec": "\n{input}",
    "triviaqa": "\n\n{input}",
    "samsum": "\n\n{input}",
    "lsht": "\n{input}",
    "passage_count": "\n\n{input}", # fake placeholder
    "passage_retrieval_en": "\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n",
    "passage_retrieval_zh": "\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n",
    "lcc": "{input}", # fake placeholder
    "repobench-p": "{input}"
}

answer_prefix = {
    "narrativeqa": "Answer:",
    "qasper": "Answer:",
    "multifieldqa_en": "Answer:",
    "multifieldqa_zh": "回答：",
    "hotpotqa": "Answer:",
    "2wikimqa": "Answer:",
    "musique": "Answer:",
    "dureader": "回答：",
    "gov_report": "Summary:",
    "qmsum": "Answer:",
    "multi_news": "Summary:",
    "vcsum": "会议总结：",
    "trec": "",
    "triviaqa": "",
    "samsum": "",
    "lsht": "",
    "passage_count": "The final answer is: ",
    "passage_retrieval_en": "The answer is: ",
    "passage_retrieval_zh": "答案是：",
    "lcc": "Next line of code:\n",
    "repobench-p": "Next line of code:\n"
}

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

datasets_e = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

tasks = datasets + [data+"_e" for data in datasets_e]

for task in tasks:
    df = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True).to_pandas()

    if task.endswith("_e"):
        task_key = task[:-2]
    else:
        task_key = task
    df = df.rename(columns={"answers": "answer", "input": "question", "dataset": "task"})
    df["context"] = df["context"].apply(lambda x: context_prompt[task_key].format(context=x))
    df["question"] = df["question"].apply(lambda x: question_prompt[task_key].format(input=x))
    df["answer_prefix"] = answer_prefix[task_key]
    df = df[["context", "question", "answer_prefix", "answer", "task", "all_classes", "length"]]
    df["max_new_tokens"] = dataset2maxlen[task_key]

    # Push to hub
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("figuremout/LongBench", config_name=task, split="test")