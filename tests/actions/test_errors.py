import unittest
from unittest.mock import patch

from lifeguard import PROBLEM
from lifeguard.validations import ValidationResponse

from lifeguard_openai.actions.errors import explain_error


class TestErrors(unittest.TestCase):
    @patch("lifeguard_openai.actions.errors.execute_prompt")
    def test_explain_error(self, mock_execute_prompt):
        validation_response = ValidationResponse(PROBLEM, {"traceback": "traceback"})
        mock_execute_prompt.return_value = "prompt response"

        explain_error(validation_response, {})
        mock_execute_prompt.assert_called_with(
            "Can you explain the root cause for the following error?\n\ntraceback"
        )
        self.assertEqual(
            validation_response.details["explanation"],
            "prompt response",
        )

    @patch("lifeguard_openai.actions.errors.execute_prompt")
    def test_explain_error_without_traceback(self, mock_execute_prompt):
        validation_response = ValidationResponse(PROBLEM, {})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], "No traceback available"
        )
        mock_execute_prompt.Completion.create.assert_not_called()

    @patch("lifeguard_openai.actions.errors.execute_prompt")
    def test_explain_error_without_choices(self, mock_execute_prompt):
        mock_execute_prompt.return_value = ""

        validation_response = ValidationResponse(PROBLEM, {"traceback": "traceback"})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], "No explanation available"
        )

    @patch("lifeguard_openai.actions.errors.execute_prompt")
    def test_explain_error_from_traceback_list(self, mock_execute_prompt):
        mock_execute_prompt.return_value = ""

        validation_response = ValidationResponse(PROBLEM, {"traceback": ["traceback"]})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], ["No explanation available"]
        )

    @patch("lifeguard_openai.actions.errors.execute_prompt")
    def test_expection_on_execute_prompt(self, mock_execute_prompt):
        mock_execute_prompt.side_effect = Exception("error")

        validation_response = ValidationResponse(PROBLEM, {"traceback": ["traceback"]})

        explain_error(validation_response, {})
        self.assertEqual(
            validation_response.details["explanation"], ["Error on explain error"]
        )
