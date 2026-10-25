# Evaluation summary  (model: gemini-3.1-flash-lite, input field: `question`)


## Multi-turn conversations (practice/conversations.csv)

- **rag**: turns 83/96 (86%); conversations fully correct 11/23; tool calls 94/96; escalations 4/6; retrieval recall 0.964
  - by scenario: {'high_value_damage': '4/4', 'return_eligibility': '7/10', 'human_requested': '7/8', 'repeat_contact': '7/8', 'refund_status': '7/8', 'where_is_my_order': '8/8', 'topic_switch': '9/10', 'compensation_demand': '6/8', 'cancel_unshipped_order': '7/8', 'cancel_shipped_order': '6/8', 'failed_verification': '7/8', 'express_late_credit': '8/8'}
  - failed turns: ['PC003#4', 'PC003#5', 'PC004#4', 'PC006#4', 'PC007#4', 'PC010#4', 'PC013#4', 'PC015#4', 'PC016#4', 'PC017#2', 'PC019#4', 'PC020#4', 'PC021#4']
