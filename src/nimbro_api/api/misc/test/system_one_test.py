from nimbro_api.utility.misc import assert_type_value, assert_log
from ..client.system_one import SystemOne

def test_1_inference():
    client = SystemOne(endpoint="OpenRouter")

    success, message, response = client.evaluate(
        state="I was charged twice for my subscription. Please refund the duplicate charge.",
        questions={
            'refund': {
                'type': "noul",
                'instructions': "Is the customer requesting a refund?"
            }
        }
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=response, type_or_value=dict, name="result of evaluate()")
    answer = response['answers']['refund']
    assert_type_value(obj=answer['type'], type_or_value="noul", name="answer type")
    assert_type_value(obj=answer['noul'], type_or_value=[float, int], name="refund probability")
    assert_log(expression=0 <= answer['noul'] <= 1, message="Expected refund probability to be between 0 and 1.")

def test_2_inference():
    client = SystemOne(endpoint="OpenRouter")

    success, message, response = client.evaluate(
        state={'ticket': "Checkout is down for all customers and no payments can be made.", 'customer_tier': "enterprise"},
        questions={
            'bug': {
                'type': "noul",
                'instructions': "Does the ticket describe broken product behavior?",
                'criteria': {'true': "Broken or unexpected behavior", 'false': "A question or feature request"}
            },
            'team': {
                'type': "choice",
                'instructions': "Which team should handle this ticket?",
                'criteria': {'payments': "Checkout and payment issues", 'account': "Login and profile issues", 'sales': "Pricing and upgrades"}
            },
            'urgency': {
                'type': "score",
                'instructions': "How urgently should the ticket be handled?",
                'criteria': ["Can wait for the next release", "Should be fixed this week", "Blocking revenue right now"]
            }
        }
    )
    assert_type_value(obj=success, type_or_value=bool, name="success")
    assert_type_value(obj=message, type_or_value=str, name="message")
    assert_log(expression=success, message=message)
    assert_type_value(obj=response, type_or_value=dict, name="result of evaluate()")
    answers = response['answers']
    assert_type_value(obj=answers, type_or_value=dict, name="answers")
    assert_type_value(obj=answers['bug']['type'], type_or_value="noul", name="bug answer type")
    assert_type_value(obj=answers['bug']['noul'], type_or_value=[float, int], name="bug probability")
    assert_log(expression=0 <= answers['bug']['noul'] <= 1, message="Expected bug probability to be between 0 and 1.")
    assert_type_value(obj=answers['team']['type'], type_or_value="choice", name="team answer type")
    assert_type_value(obj=answers['team']['choice'], type_or_value=['payments', 'account', 'sales'], name="selected team")
    assert_type_value(obj=answers['team']['probabilities'], type_or_value=dict, name="team probabilities")
    assert_type_value(obj=answers['team']['confidence'], type_or_value=[float, int], name="team confidence")
    assert_type_value(obj=answers['urgency']['type'], type_or_value="score", name="urgency answer type")
    assert_type_value(obj=answers['urgency']['score'], type_or_value=[float, int], name="urgency score")
    assert_log(expression=0 <= answers['urgency']['score'] <= 2, message="Expected urgency score to be between 0 and 2.")
    assert_type_value(obj=answers['urgency']['probabilities'], type_or_value=dict, name="urgency probabilities")
    assert_type_value(obj=answers['urgency']['legend'], type_or_value=dict, name="urgency legend")
    assert_type_value(obj=response['model'], type_or_value=str, name="response model")
    assert_type_value(obj=response['usage'], type_or_value=dict, name="response usage")
