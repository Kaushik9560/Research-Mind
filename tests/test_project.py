"""Regression checks for pipeline wiring, evidence parsing, and UI interactions."""

import io
import os
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import agents
import pipeline
import tools
from dry_run import run_dry_run


class EvidenceTests(unittest.TestCase):
    def test_clean_text_and_abstract_positions(self):
        self.assertEqual(tools.clean_text('<p>AI &amp; science</p>\n today'), 'AI & science today')
        self.assertEqual(tools.clean_text(None), '')
        self.assertEqual(tools.clean_text('abcdef', 3), 'abc')
        self.assertEqual(tools.rebuild_abstract({'works': [2], 'AI': [0, 3], 'tutoring': [1]}),
                         'AI tutoring works AI')
        self.assertEqual(tools.rebuild_abstract(None), '')

    @patch('tools.search_news')
    @patch('tools.search_openalex')
    def test_deduplicates_and_assigns_ids_without_changing_originals(self, academic, news):
        first = {'title': 'First', 'url': 'https://example.com/a'}
        second = {'title': 'Second', 'url': 'https://example.com/A'}
        academic.return_value = [first, second]
        news.return_value = [first.copy(), {'title': 'No URL'}, {'title': 'no url'}, {}]
        evidence, warnings = tools.collect_evidence('AI', 30, 4)
        self.assertEqual([item['id'] for item in evidence], ['S1', 'S2', 'S3'])
        self.assertNotIn('id', first)
        self.assertNotIn('id', second)
        self.assertEqual(warnings, [])
        academic.assert_called_once_with('AI', 30, 4)
        news.assert_called_once_with('AI', 30, 4)

    @patch('tools.search_news', return_value=[{'title': 'News', 'url': 'https://example.com'}])
    @patch('tools.search_openalex', side_effect=requests.Timeout)
    def test_failed_provider_keeps_successful_results(self, academic, news):
        evidence, warnings = tools.collect_evidence('AI', 30, 4)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(warnings, ['OpenAlex could not be reached.'])

    @patch.dict(os.environ, {'TAVILY_API_KEY': ''})
    def test_missing_optional_key_is_visible(self):
        evidence, warnings = tools.collect_evidence('AI', 30, 4, False, False, True)
        self.assertEqual(evidence, [])
        self.assertIn('TAVILY_API_KEY', warnings[0])
        self.assertIn('NO LIVE EVIDENCE', tools.format_evidence(evidence))

    @patch('tools.requests.get')
    def test_invalid_news_xml_is_a_warning(self, get):
        get.return_value.content = b'not XML'
        evidence, warnings = tools.collect_evidence('AI', 30, 4, False, True)
        self.assertEqual(evidence, [])
        self.assertEqual(warnings, ['Google News could not be reached.'])

    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test-only'})
    @patch('tools.requests.post')
    def test_tavily_receives_requested_date_window(self, post):
        post.return_value.json.return_value = {'results': [{'title': 'Page', 'content': '<p>Text</p>'}]}
        records = tools.search_tavily('AI', 30, 4)
        body = post.call_args.kwargs['json']
        self.assertEqual(body['start_date'], (date.today() - timedelta(days=30)).isoformat())
        self.assertEqual(body['end_date'], date.today().isoformat())
        self.assertEqual(records[0]['summary'], 'Text')

    @patch.dict(os.environ, {'OPENALEX_API_KEY': 'test-only'})
    @patch('tools.requests.get')
    def test_openalex_optional_key_and_missing_metadata(self, get):
        get.return_value.json.return_value = {'results': [{'primary_location': None}]}
        records = tools.search_openalex('AI', 30, 4)
        self.assertEqual(get.call_args.kwargs['headers']['Authorization'], 'Bearer test-only')
        self.assertEqual(records[0]['publisher'], 'OpenAlex')
        self.assertEqual(records[0]['summary'], '')


class ModelTests(unittest.TestCase):
    @patch.dict(os.environ, {'GOOGLE_API_KEY': ''})
    def test_missing_research_key_has_clear_error(self):
        with self.assertRaisesRegex(agents.ModelConfigurationError, 'GOOGLE_API_KEY'):
            agents.create_models()

    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-only', 'MAX_OUTPUT_TOKENS': 'invalid'})
    def test_invalid_token_setting_has_clear_error(self):
        with self.assertRaisesRegex(agents.ModelConfigurationError, 'positive integer'):
            agents.create_models()

    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-only', 'GROQ_API_KEY': '', 'MAX_OUTPUT_TOKENS': '1400'})
    @patch('agents.ChatGoogleGenerativeAI')
    @patch('agents.ChatGroq')
    def test_critic_reuses_gemini_when_groq_is_absent(self, groq, google):
        research, critic = agents.create_models()
        self.assertIs(research, critic)
        groq.assert_not_called()

    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-only', 'GROQ_API_KEY': 'test-only', 'MAX_OUTPUT_TOKENS': '1400'})
    @patch('agents.ChatGoogleGenerativeAI')
    @patch('agents.ChatGroq')
    def test_groq_is_used_when_configured(self, groq, google):
        research, critic = agents.create_models()
        self.assertIs(research, google.return_value)
        self.assertIs(critic, groq.return_value)

    def test_followup_receives_context_question_and_language(self):
        prompts = []
        def reply(prompt):
            prompts.append(prompt.to_messages()[1].content)
            return AIMessage(content='Supported answer [S1]')
        model = RunnableLambda(reply)
        with patch('agents.create_models', return_value=(model, model)):
            answer = agents.answer_follow_up('Evidence [S1]', 'What changed?', 'Hindi')
        self.assertEqual(answer, 'Supported answer [S1]')
        self.assertIn('Evidence [S1]', prompts[0])
        self.assertIn('What changed?', prompts[0])
        self.assertIn('Hindi', prompts[0])


class PipelineTests(unittest.TestCase):
    def test_query_removes_fillers_and_duplicate_words(self):
        self.assertEqual(pipeline.make_search_query('What are the latest AI AI trends?', 'General', 'Global'), 'ai')
        self.assertEqual(pipeline.make_search_query('AI', 'Education', 'India'), 'ai education india')
        self.assertEqual(pipeline.make_search_query('the', 'General', 'Global'), 'the')
        topic = ' '.join(['AI'] * 15 + ['education'])
        self.assertEqual(pipeline.make_search_query(topic, 'General', 'Global'), 'ai education')

    @patch('pipeline.create_models')
    def test_invalid_requests_do_not_make_model_calls(self, create):
        for options in (
            {'topic': ' '}, {'topic': 'AI', 'days': 0}, {'topic': 'AI', 'depth': 'Unknown'},
            {'topic': 'AI', 'use_academic': False, 'use_news': False, 'use_web': False},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pipeline.run_research_pipeline(**options)
        create.assert_not_called()

    def test_real_pipeline_passes_outputs_to_next_stage(self):
        prompts = []
        outputs = ['PLAN', 'NOTES [S1]', 'REPORT [S1]', 'REVIEW']
        events = []
        def reply(prompt):
            prompts.append(prompt.to_messages()[1].content)
            return AIMessage(content=outputs[len(prompts) - 1])
        model = RunnableLambda(reply)
        source = {'id': 'S1', 'title': 'Paper', 'kind': 'Academic', 'summary': 'Finding'}
        with (
            patch('pipeline.create_models', return_value=(model, model)),
            patch('pipeline.collect_evidence', return_value=([source], ['Demo warning'])) as collect,
        ):
            result = pipeline.run_research_pipeline('AI education', on_progress=lambda *args: events.append(args))
        self.assertEqual(len(prompts), 4)
        self.assertIn('PLAN', prompts[1])
        self.assertIn('NOTES [S1]', prompts[2])
        self.assertIn('REPORT [S1]', prompts[3])
        self.assertIn('[S1] Paper', prompts[1])
        self.assertEqual(result['reader_notes'], outputs[1])
        self.assertEqual(events, [(stage, status) for stage in ('search', 'reader', 'writer', 'critic')
                                  for status in ('running', 'done')])
        self.assertEqual(collect.call_args.kwargs['per_source'], 6)
        exported = pipeline.export_markdown(result)
        for value in ('PLAN', 'NOTES [S1]', 'REPORT [S1]', 'REVIEW', 'Demo warning', 'Finding'):
            self.assertIn(value, exported)

    def test_no_evidence_warning_reaches_reader_writer_and_critic(self):
        prompts = []
        def reply(prompt):
            prompts.append(prompt.to_messages()[1].content)
            return AIMessage(content='No evidence available.')
        model = RunnableLambda(reply)
        with (
            patch('pipeline.create_models', return_value=(model, model)),
            patch('pipeline.collect_evidence', return_value=([], ['No source available'])),
        ):
            result = pipeline.run_research_pipeline('AI')
        self.assertEqual(result['evidence'], [])
        for prompt in prompts[1:]:
            self.assertIn('NO LIVE EVIDENCE', prompt)

    def test_offline_demo_never_calls_external_models_or_post(self):
        with (
            patch('agents.ChatGoogleGenerativeAI', side_effect=AssertionError('Live model called')),
            patch('tools.requests.post', side_effect=AssertionError('Network POST called')),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = run_dry_run()
        self.assertEqual([source['id'] for source in result['evidence']], ['S1', 'S2'])
        self.assertEqual(result['evidence'][0]['summary'], 'AI tutoring may support practice.')
        self.assertEqual(result['warnings'], [])
        self.assertIn('OFFLINE DRY RUN', output.getvalue())
        self.assertIn('FINAL RESULT DICTIONARY', output.getvalue())


class AppTests(unittest.TestCase):
    def test_app_loads_and_empty_question_is_rejected(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        self.assertEqual(len(app.exception), 0)
        app.button[-1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('Enter a research question', app.warning[0].value)

    def test_research_results_and_followups_render(self):
        with redirect_stdout(io.StringIO()):
            result = run_dry_run()
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.text_area[0].set_value('AI education')
        with patch('pipeline.run_research_pipeline', return_value=result) as run:
            app.button[-1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 5)
        self.assertEqual(app.session_state['research_result']['report'], result['report'])
        self.assertEqual(run.call_args.kwargs['topic'], 'AI education')
        with patch('agents.answer_follow_up', return_value='Sample follow-up [S1]') as answer:
            app.chat_input[0].set_value('What does S1 say?').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state['followups']), 2)
        self.assertIn('[S1]', answer.call_args.args[0])
        self.assertEqual(app.session_state['followups'][-1]['content'], 'Sample follow-up [S1]')

    def test_example_button_fills_question(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('Small language models', app.text_area[0].value)

    def test_failed_run_preserves_previous_report_and_chat(self):
        with redirect_stdout(io.StringIO()):
            result = run_dry_run()
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.session_state['research_result'] = result
        history = [{'role': 'assistant', 'content': 'Existing answer'}]
        app.session_state['followups'] = history
        app.text_area[0].set_value('New research')
        with patch('pipeline.run_research_pipeline', side_effect=RuntimeError('503 UNAVAILABLE: high demand')):
            app.button[-1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['research_result'], result)
        self.assertEqual(app.session_state['followups'], history)
        self.assertTrue(app.error)
        self.assertIn('provider is busy', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
