import unittest
import pandas as pd  # type: ignore
from prepro import prepro, clean_text, lemmatize_text, tokenize_text, transform_label


class TestPrepro(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'title': ['Help needed', None],
            'text': ['I feel so lost...', 'No one understands me.'],
            'is_suicide': ['no', 'yes']
        })

        # Test cases for individual functions
        self.test_text = "I'm feeling very sad today... Don't know what to do!"
        self.test_text_with_contractions = "I'll be there, don't worry, we're going, they've left, it's time"
        self.test_text_with_special_chars = "Hello! How are you? I'm fine... #hashtag @mention http://url.com"
        self.test_text_with_stopwords = "the quick brown fox jumps over the lazy dog"
        self.test_text_with_numbers = "I have 123 problems but 456 solutions"
        self.test_text_with_emojis = "I'm happy 😊 but also sad 😢"
        self.test_text_with_html = "<p>This is <b>bold</b> and <i>italic</i></p>"
        self.test_text_with_unicode = "café résumé naïve"
        self.test_text_with_repeated_chars = "I'm soooooo happy!!!!!"
        self.test_text_with_mixed_case = "HeLLo WoRlD"
        self.test_text_with_extra_spaces = "  too   many    spaces   "
        self.test_text_with_punctuation = "Hello, world! How are you? I'm fine; thank you."

    def test_clean_text(self):
        """Test the clean_text function"""
        # Test basic cleaning
        cleaned = clean_text(self.test_text)
        self.assertTrue(cleaned.islower())
        self.assertFalse(any(char in cleaned for char in ['.', ',', '!', '?']))

        # Test special character removal
        cleaned = clean_text(self.test_text_with_special_chars)
        self.assertFalse(any(char in cleaned for char in [
                         '!', '?', '.', '#', '@', ':', '/']))

        # Test number removal
        cleaned = clean_text(self.test_text_with_numbers)
        self.assertFalse(any(char.isdigit() for char in cleaned))

        # Test emoji removal
        cleaned = clean_text(self.test_text_with_emojis)
        self.assertFalse(any(ord(char) > 127 for char in cleaned))

        # Test HTML removal
        cleaned = clean_text(self.test_text_with_html)
        self.assertNotIn('<p>', cleaned)
        self.assertNotIn('<b>', cleaned)
        self.assertNotIn('<i>', cleaned)

        # Test case conversion
        cleaned = clean_text(self.test_text_with_mixed_case)
        self.assertEqual(cleaned, 'hello world')

        # Test punctuation removal
        cleaned = clean_text(self.test_text_with_punctuation)
        self.assertFalse(any(char in cleaned for char in [',', '!', '?', ';']))

    def test_lemmatize_text(self):
        """Test the lemmatize_text function"""
        # Test basic lemmatization
        lemmatized = lemmatize_text("running runs ran")
        self.assertIn('run', lemmatized)

        # Test with different word forms
        lemmatized = lemmatize_text("better best good")
        self.assertIn('good', lemmatized)

        # Test with empty string
        self.assertEqual(lemmatize_text(""), "")

        # Test with irregular verbs
        lemmatized = lemmatize_text("went going go")
        self.assertIn('go', lemmatized)

        # Test with plural forms
        lemmatized = lemmatize_text("cats dogs mice")
        self.assertIn('cat', lemmatized)
        self.assertIn('dog', lemmatized)
        self.assertIn('mouse', lemmatized)

        # Test with adjectives
        lemmatized = lemmatize_text("happier happiest happy")
        self.assertIn('happy', lemmatized)

    def test_tokenize_text(self):
        """Test the tokenize_text function"""
        # Test basic tokenization
        tokens = tokenize_text("hello world")
        self.assertEqual(tokens, ['hello', 'world'])

        # Test with multiple spaces
        tokens = tokenize_text("hello   world")
        self.assertEqual(tokens, ['hello', 'world'])

        # Test with empty string
        self.assertEqual(tokenize_text(""), [])

        # Test with tabs and newlines
        tokens = tokenize_text("hello\tworld\npython")
        self.assertEqual(tokens, ['hello', 'world', 'python'])

        # Test with leading/trailing spaces
        tokens = tokenize_text("  hello world  ")
        self.assertEqual(tokens, ['hello', 'world'])

        # Test with special characters (should be cleaned first)
        tokens = tokenize_text(clean_text("Hello, World!"))
        self.assertEqual(tokens, ['hello', 'world'])

    def test_transform_label(self):
        """Test the transform_label function"""
        # Test 'yes' case
        self.assertEqual(transform_label('yes'), 1)

        # Test 'no' case
        self.assertEqual(transform_label('no'), 0)

        # Test invalid input
        self.assertEqual(transform_label('invalid'), 0)

        # Test case sensitivity
        self.assertEqual(transform_label('yes'), 1)
        self.assertEqual(transform_label('No'), 0)

        # Test whitespace handling
        self.assertEqual(transform_label('yes'), 1)
        self.assertEqual(transform_label(' no '), 0)

        # Test empty string
        self.assertEqual(transform_label(''), 0)

        # Test None
        self.assertEqual(transform_label(None), 0)

    def test_prepro_output(self):
        """Test the main prepro function"""
        result_df = prepro(self.df)
        # Check columns
        self.assertIn('full_text', result_df.columns)
        self.assertIn('text_clean', result_df.columns)
        self.assertIn('tokens', result_df.columns)

        # Check text cleaning
        for text in result_df['text_clean']:
            self.assertFalse(
                any(char in text for char in ['.', ',', '!', '?']))

        # Check label transformation
        self.assertEqual(result_df['is_suicide'].dtype, 'int64')
        self.assertIn(0, result_df['is_suicide'].values)
        self.assertIn(1, result_df['is_suicide'].values)

        # Check tokenization
        for tokens in result_df['tokens']:
            self.assertIsInstance(tokens, list)
            self.assertTrue(all(isinstance(token, str) for token in tokens))

        # Check handling of None values
        # Should handle None in title
        self.assertIsNotNone(result_df['full_text'].iloc[1])

        # Check that all text is properly cleaned
        for text in result_df['text_clean']:
            self.assertFalse(any(char.isdigit() for char in text))
            # No non-ASCII characters
            self.assertFalse(any(ord(char) > 127 for char in text))


if __name__ == '__main__':
    unittest.main()
