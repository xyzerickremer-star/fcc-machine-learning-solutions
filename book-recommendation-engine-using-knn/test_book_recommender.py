"""Local smoke tests for the FCC KNN book recommender."""

from typing import cast

from book_recommender import Recommendation, get_recommends


def test_fcc_shape_and_known_query():
    query = "Where the Heart Is (Oprah's Book Club (Paperback))"
    result = get_recommends(query)

    recommendations = cast(list[Recommendation], result[1])

    expected_titles = [
        "The Lovely Bones: A Novel",
        "I Know This Much Is True",
        "The Surgeon",
        "The Weight of Water",
        "I'll Be Seeing You",
    ]

    assert result[0] == query
    assert [title for title, _ in recommendations] == expected_titles
    assert len(recommendations) == 5
    assert all(isinstance(title, str) for title, _ in recommendations)
    assert all(isinstance(distance, float) for _, distance in recommendations)
    assert query not in [title for title, _ in recommendations]


def test_recommendations_are_sorted_by_nearest_distance():
    result = get_recommends(
        "The Queen of the Damned (Vampire Chronicles (Paperback))"
    )
    recommendations = cast(list[Recommendation], result[1])
    distances = [distance for _, distance in recommendations]

    assert distances == sorted(distances)
