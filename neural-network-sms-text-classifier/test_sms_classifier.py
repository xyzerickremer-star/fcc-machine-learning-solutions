"""Local checks mirroring the FCC SMS classifier sample tests."""

from sms_classifier import predict_message


TEST_MESSAGES = [
    "how are you doing today",
    "sale today! to stop texts call 98912460 4 08712460324",
    "i dont want to go. can we try it a different day? available sat",
    "our new mobile video service is live. just install on your phone to start watching.",
    "you have won £1000 cash! call to claim your prize.",
    "i'll bring it tomorrow. don't forget the milk.",
    "wow, is your arm alright. that happened to me one time too",
]

TEST_ANSWERS = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]


def test_fcc_sample_messages():
    predictions = [predict_message(message)[1] for message in TEST_MESSAGES]
    assert predictions == TEST_ANSWERS


if __name__ == "__main__":
    for message, expected in zip(TEST_MESSAGES, TEST_ANSWERS):
        probability, label = predict_message(message)
        print(f"{label:4s} p_spam={probability:.4f} expected={expected:4s} | {message}")
        assert label == expected
    print("All FCC sample SMS tests passed.")
