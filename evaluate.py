import os

# --- Safety net for Python 3.13/3.14 -------------------------------------
# RAGAS calls nest_asyncio.apply() when it is imported. nest_asyncio breaks
# asyncio on newer Pythons, which makes every metric job fail with
# "RuntimeError: Timeout should be used inside a task" and every score come
# back as NaN. A plain script has no running event loop, so nest_asyncio is
# not needed here. This line must stay ABOVE the ragas imports.
import nest_asyncio
nest_asyncio.apply = lambda *args, **kwargs: None
# --------------------------------------------------------------------------

from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.run_config import RunConfig

from rag_chain import rag_chain, retriever

load_dotenv()

# Each test case needs a "reference": the ground-truth answer, written from
# what is actually stated in YOUR report files.
test_cases = [
    {
        "question": "How did NexGen Systems' email campaign for women aged 35-44 in New York perform?",
        "reference": "It was one of the strongest campaigns of the quarter. The 60-day email campaign converted at 12%, roughly double the email average, at an acquisition cost of about $11,500 and an ROI of 5.6x. Engagement scored 7 out of 10."
    },
    {
        "question": "Why did Innovate Industries' influencer campaign for the Foodies segment underperform?",
        "reference": "The influencer content and the landing experience it was routed to (Google Ads) had almost nothing in common. The Mandarin-language creative was reportedly a direct translation of the English script with no localization, which likely hurt credibility. The campaign got 659 clicks from about 9,000 impressions, converted at only 5%, and scored 1 out of 10 on engagement."
    },
    {
        "question": "What was the ROI of Innovate Industries' influencer campaign, and how much did it cost?",
        "reference": "It cost about $17,500 over 60 days and returned an ROI of 3.6x. That is not a disaster on paper, but it was well below what the same budget earned on email."
    },
    {
        "question": "What went wrong with DataTech Solutions' 15-day Instagram display campaign in New York?",
        "reference": "It targeted Foodies and produced only 100 clicks from 1,643 impressions, with an engagement score of 1. The likely cause was the short 15-day duration, which was not enough for the algorithm to optimize."
    },
    {
        "question": "What is the minimum campaign duration recommended for display ads, and why?",
        "reference": "The DataTech retrospective concluded that display needs at least 45 days of runway. Shorter runs should be treated as test budgets rather than real campaigns, because the algorithm does not have enough time to optimize."
    },
    {
        "question": "How did DataTech's Miami display campaign perform?",
        "reference": "The 60-day YouTube display campaign, targeting all age groups and reaching the Health & Wellness segment, converted at 11% with an ROI of 5.55. The team considered it solid."
    },
    {
        "question": "Why did stakeholders question the success of Alpha Innovations' Facebook campaign in Chicago?",
        "reference": "The 15-day campaign cost $18,000, among the highest acquisition costs in the dataset. It got 861 clicks from fewer than 1,800 impressions, but conversion stayed at 9%. An analyst said the audience was interested but the checkout process lost them. The ROI of 6.7x means it technically paid off, but stakeholders questioned whether that was luck or repeatable performance."
    },
    {
        "question": "Which TechCorp campaign performed best and which was cancelled?",
        "reference": "The 45-day email campaign for men 25-34 in New York performed best, with 14% conversion, $9,975 acquisition cost and a 7.06 ROI. The Houston display campaign was cancelled after three weeks when click volume flatlined."
    },
    {
        "question": "How did TechCorp's Facebook campaign compare to its email campaign for men 25-34?",
        "reference": "The Facebook campaign scored 10 on engagement but converted at only 9%, suggesting people liked the content but did not buy. The email campaign converted at 14% with a 7.06 ROI, and email did most of the heavy lifting for TechCorp that quarter."
    },
    {
        "question": "Which campaigns show that high engagement does not guarantee conversions?",
        "reference": "TechCorp's Facebook campaign for men 25-34 had an engagement score of 10 but only 9% conversion. Alpha Innovations' Facebook campaign in Chicago had a very high click volume (861 clicks from under 1,800 impressions) but also converted at 9%, with an analyst noting that checkout lost interested users."
    },
    {
        "question": "What relationship was observed between acquisition cost and campaign performance?",
        "reference": "Campaigns with acquisition costs under $10,000 tended to outperform those above $15,000, and the relationship was stronger than the team expected."
    },
    {
        "question": "Which languages performed best on engagement?",
        "reference": "French and German campaigns outperformed English and Spanish on engagement, possibly because competition for attention was lower. The NexGen report also noted that German-language email variants beat the English control."
    },
    {
        "question": "How does campaign duration relate to performance, and is there an exception?",
        "reference": "Longer campaigns of 45-60 days generally beat shorter ones. The exception is email, where a well-timed 15-day burst often matched the 60-day results at a fraction of the cost."
    },
    {
        "question": "Which cities delivered better ROI per dollar?",
        "reference": "Houston and Miami produced better ROI per dollar than Los Angeles. The sample sizes were small enough that no one wanted to draw firm conclusions."
    },
    {
        "question": "What action did leadership request after NexGen's email campaign?",
        "reference": "Leadership asked that the German-language creative be reused in the Chicago rollout."
    },
    {
        "question": "What was the conversion rate of Microsoft's search campaign in Tokyo?",
        "reference": "The reports do not contain any information about a Microsoft search campaign in Tokyo."
    },
]

# Run each question through your RAG pipeline
results = {"question": [], "answer": [], "contexts": [], "reference": []}
for i, case in enumerate(test_cases, 1):
    q = case["question"]
    print(f"[{i}/{len(test_cases)}] {q}")
    docs = retriever.invoke(q)
    results["question"].append(q)
    results["answer"].append(rag_chain.invoke(q))
    results["contexts"].append([d.page_content for d in docs])
    results["reference"].append(case["reference"])

dataset = Dataset.from_dict(results)

# Explicit judge model + embeddings for RAGAS (instead of relying on defaults)
judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

# max_workers=4 keeps you under OpenAI rate limits on a new account
run_config = RunConfig(timeout=180, max_workers=4)

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=run_config,
)

df = scores.to_pandas()
df.to_csv("ragas_scores.csv", index=False)

print("\n=== Average scores ===")
print(df[["faithfulness", "answer_relevancy", "context_precision"]].mean(numeric_only=True))

# Guard against silently-empty results
if df[["faithfulness", "answer_relevancy", "context_precision"]].isna().all().all():
    print("\nWARNING: every score is NaN, so no metric job succeeded. "
          "Scroll up for 'Exception raised in Job' lines, and check your "
          "OPENAI_API_KEY, billing credit, and `python --version` (use 3.12).")
else:
    print("\nSaved detailed scores to ragas_scores.csv")