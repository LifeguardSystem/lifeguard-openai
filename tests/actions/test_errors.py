import unittest
from unittest.mock import patch

from lifeguard import PROBLEM
from lifeguard.validations import ValidationResponse

from lifeguard_openai.actions.errors import explain_error


class TestErrors(unittest.TestCase):
    @patch("lifeguard_openai.actions.errors.openai")
    def test_explain_error(self, mock_openai):
        validation_response = ValidationResponse(PROBLEM, {"traceback": "traceback"})

        explain_error(validation_response, {})
        mock_openai.Completion.create.assert_called_with(
            model="text-davinci-003",
            prompt="Can you explain the root cause for the following error?\n\ntraceback",
            temperature=0.0,
            top_p=0.8,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=200,
        )
        self.assertEqual(
            validation_response.details["explanation"],
            mock_openai.Completion.create.return_value.choices[0].text,
        )

    @patch("lifeguard_openai.actions.errors.openai")
    def test_explain_error_without_traceback(self, mock_openai):
        validation_response = ValidationResponse(PROBLEM, {})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], "No traceback available"
        )
        mock_openai.Completion.create.assert_not_called()

    @patch("lifeguard_openai.actions.errors.openai")
    def test_explain_error_without_choices(self, mock_openai):
        mock_openai.Completion.create.return_value.choices = []

        validation_response = ValidationResponse(PROBLEM, {"traceback": "traceback"})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], "No explanation available"
        )

    @patch("lifeguard_openai.actions.errors.openai")
    def test_explain_error_from_traceback_list(self, mock_openai):
        mock_openai.Completion.create.return_value.choices = []

        validation_response = ValidationResponse(PROBLEM, {"traceback": ["traceback"]})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], ["No explanation available"]
        )
