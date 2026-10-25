# Evaluation summary  (model: gemini-3.1-flash-lite, input field: `question`)

## Single questions (practice/questions.csv)

| Agent | Passed | Rate | Retrieval recall | Unanswerable handled | Avg latency |
|---|---|---|---|---|---|
| Step 1 prompt-only | 12/52 | 23% | - | 3/5 | 3.27s |
| Step 2 RAG agent | 39/52 | 75% | 1.0 | 2/5 | 5.79s |

**baseline by language:** {'hi-en': '1/11', 'en': '8/33', 'hi': '3/8'}

**baseline by intent:** {'warranty': '0/5', 'refund_timeline': '1/14', 'unsupported': '3/5', 'return_fee': '1/2', 'return_eligibility': '1/4', 'return_window': '3/11', 'delivery_attempts': '0/1', 'fraud': '0/1', 'damaged_item': '0/3', 'troubleshooting': '0/1', 'delivery_area': '1/1', 'cancel_order': '1/2', 'late_delivery': '0/1', 'policy_override': '1/1'}

**rag by language:** {'hi-en': '6/11', 'en': '28/33', 'hi': '5/8'}

**rag by intent:** {'warranty': '3/5', 'refund_timeline': '12/14', 'unsupported': '2/5', 'return_fee': '2/2', 'return_eligibility': '4/4', 'return_window': '8/11', 'delivery_attempts': '1/1', 'fraud': '0/1', 'damaged_item': '1/3', 'troubleshooting': '1/1', 'delivery_area': '1/1', 'cancel_order': '2/2', 'late_delivery': '1/1', 'policy_override': '1/1'}
