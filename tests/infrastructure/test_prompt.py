import unittest
from unittest.mock import patch

from lifeguard_openai.infrastructure.prompt import execute_prompt


class TestErrors(unittest.TestCase):
    @patch("lifeguard_openai.infrastructure.prompt.openai")
    def test_execute_prompt(self, mock_openai):
        response = execute_prompt("prompt")

        mock_openai.ChatCompletion.create.assert_called_with(
            temperature=0.0,
            top_p=0.8,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=200,
            model="text-davinci-003",
            messages=[{"role": "User", "content": "prompt"}],
            stop=[" User:", " Assistant:"],
        )
        self.assertEqual(
            response,
            mock_openai.ChatCompletion.create.return_value.choices[0].message.content,
        )

    @patch("lifeguard_openai.infrastructure.prompt.openai")
    def test_execute_prompt_without_response(self, mock_openai):
        mock_openai.ChatCompletion.create.return_value.choices = []

        response = execute_prompt("prompt")

        self.assertEqual(response, "")
