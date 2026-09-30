import json
import math
import time

from nimbro_api.client import ClientBase
from nimbro_api.utility.api import get_api_key, validate_endpoint, post_request
from nimbro_api.utility.misc import UnrecoverableError, assert_type_value, assert_keys, assert_log, escape

class SystemOneBase(ClientBase):

    def __init__(self, settings, default_settings, **kwargs):
        super().__init__(settings=settings, default_settings=default_settings, **kwargs)
        self.get_api_key = get_api_key.__get__(self)
        self._logger.debug(f"Initialized '{type(self).__name__}' object.")
        self._initialized = True

    def set_settings(self, settings, mode="set"):
        settings = self._introduce_settings(settings=settings, mode=mode)

        # message_results
        assert_type_value(obj=settings['message_results'], type_or_value=bool, name="setting 'message_results'")

        # endpoints
        assert_type_value(obj=settings['endpoints'], type_or_value=dict, name="setting 'endpoints'")
        assert_log(expression=len(settings['endpoints']) > 0, message="Expected setting 'endpoints' to define at least one endpoint.")
        for endpoint in settings['endpoints']:
            assert_type_value(obj=endpoint, type_or_value=str, name="all endpoint names in setting 'endpoints'")
            assert_log(expression=len(endpoint) > 0, message="Expected all endpoint names in setting 'endpoints' to be non-empty.")
            validate_endpoint(endpoint=settings['endpoints'][endpoint], flavors=None, require_key=True, require_name=False, allow_models=False, setting_name=f"endpoint '{endpoint}' in setting 'endpoints'")

        # endpoint
        if isinstance(settings['endpoint'], dict):
            validate_endpoint(endpoint=settings['endpoint'], flavors=None, require_key=True, require_name=True, allow_models=False, setting_name="endpoint provided through setting 'endpoint'")
            settings['endpoints'][settings['endpoint']['name']] = settings['endpoint']
            settings['endpoint'] = settings['endpoint']['name']
            del settings['endpoints'][settings['endpoint']]['name']
        else:
            assert_type_value(obj=settings['endpoint'], type_or_value=list(settings['endpoints'].keys()), name="setting 'endpoint'")

        # model
        assert_type_value(obj=settings['model'], type_or_value=[str, None], name="setting 'model'")
        if settings['model'] is not None:
            assert_log(expression=len(settings['model']) > 0, message="Expected setting 'model' to be None or a non-empty string.")

        # timeout_connect
        assert_type_value(obj=settings['timeout_connect'], type_or_value=[float, int, None], name="setting 'timeout_connect'")
        if settings['timeout_connect'] is not None:
            assert_log(expression=settings['timeout_connect'] > 0.0, message=f"Expected setting 'timeout_connect' to be None or greater than zero but got '{settings['timeout_connect']}'.")

        # timeout_read
        assert_type_value(obj=settings['timeout_read'], type_or_value=[float, int, None], name="setting 'timeout_read'")
        if settings['timeout_read'] is not None:
            assert_log(expression=settings['timeout_read'] > 0.0, message=f"Expected setting 'timeout_read' to be None or greater than zero but got '{settings['timeout_read']}'.")

        # apply settings
        self._endpoint = settings['endpoints'][settings['endpoint']]
        return self._apply_settings(settings, mode)

    def _validate_questions(self, questions):
        assert_type_value(obj=questions, type_or_value=dict, name="argument 'questions'")
        assert_log(expression=len(questions) > 0, message="Expected argument 'questions' to contain at least one question.")
        for identifier, question in questions.items():
            name = f"question '{identifier}'"
            assert_type_value(obj=identifier, type_or_value=str, name="question identifier")
            assert_log(expression=len(identifier) > 0, message="Expected question identifiers to be non-empty strings.")
            assert_type_value(obj=question, type_or_value=dict, name=name)
            assert_keys(obj=question, keys=['type', 'instructions'], mode="required", name=name)
            assert_type_value(obj=question['type'], type_or_value=['noul', 'choice', 'score'], name=f"key 'type' in {name}")
            assert_type_value(obj=question['instructions'], type_or_value=[str, dict, list], name=f"key 'instructions' in {name}")
            if question['type'] == "noul":
                if 'criteria' in question:
                    assert_type_value(obj=question['criteria'], type_or_value=dict, name=f"key 'criteria' in {name}")
                    assert_keys(obj=question['criteria'], keys=['true', 'false'], name=f"key 'criteria' in {name}")
                    for value in question['criteria'].values():
                        assert_type_value(obj=value, type_or_value=[str, dict, list], name=f"criterion in {name}")
            else:
                assert_keys(obj=question, keys=['criteria'], mode="required", name=name)
                criteria = question['criteria']
                if question['type'] == "choice":
                    assert_type_value(obj=criteria, type_or_value=dict, name=f"key 'criteria' in {name}")
                    assert_log(expression=len(criteria) > 0, message=f"Expected criteria in {name} to contain at least one option.")
                    for label, value in criteria.items():
                        assert_type_value(obj=label, type_or_value=str, name=f"option label in {name}")
                        assert_type_value(obj=value, type_or_value=[str, dict, list, None], name=f"criterion in {name}")
                else:
                    assert_type_value(obj=criteria, type_or_value=list, name=f"key 'criteria' in {name}")
                    assert_log(expression=len(criteria) > 0, message=f"Expected criteria in {name} to contain at least one level.")
                    for value in criteria:
                        assert_type_value(obj=value, type_or_value=[str, dict, list], name=f"criterion in {name}")

    def _validate_response(self, response, questions):
        assert_type_value(obj=response, type_or_value=dict, name="API response")
        assert_keys(obj=response, keys=['answers'], mode="required", name="API response")
        assert_type_value(obj=response['answers'], type_or_value=dict, name="key 'answers' in API response")
        assert_keys(obj=response['answers'], keys=questions.keys(), mode="required", name="answers in API response")
        for identifier, question in questions.items():
            answer = response['answers'][identifier]
            name = f"answer '{identifier}' in API response"
            kind = question['type']
            assert_type_value(obj=answer, type_or_value=dict, name=name)
            assert_keys(obj=answer, keys=['type', kind], mode="required", name=name)
            assert_type_value(obj=answer['type'], type_or_value=kind, name=f"key 'type' in {name}")
            if kind == "choice":
                assert_type_value(obj=answer[kind], type_or_value=list(question['criteria'].keys()), name=f"key '{kind}' in {name}")
            else:
                assert_type_value(obj=answer[kind], type_or_value=[float, int], name=f"key '{kind}' in {name}")
                maximum = 1 if kind == "noul" else len(question['criteria']) - 1
                assert_log(expression=math.isfinite(answer[kind]) and 0 <= answer[kind] <= maximum, message=f"Expected key '{kind}' in {name} to be between 0 and {maximum}.")
            if 'confidence' in answer:
                assert_type_value(obj=answer['confidence'], type_or_value=[float, int], name=f"key 'confidence' in {name}")
                assert_log(expression=0 <= answer['confidence'] <= 1, message=f"Expected confidence in {name} to be between 0 and 1.")
            if 'probabilities' in answer:
                assert_type_value(obj=answer['probabilities'], type_or_value=dict, name=f"key 'probabilities' in {name}")
                for label, probability in answer['probabilities'].items():
                    assert_type_value(obj=label, type_or_value=str, name=f"probability label in {name}")
                    assert_type_value(obj=probability, type_or_value=[float, int], name=f"probability in {name}")
                    assert_log(expression=0 <= probability <= 1, message=f"Expected probabilities in {name} to be between 0 and 1.")
            if 'legend' in answer:
                assert_type_value(obj=answer['legend'], type_or_value=dict, name=f"key 'legend' in {name}")
                for label, value in answer['legend'].items():
                    assert_type_value(obj=label, type_or_value=str, name=f"legend label in {name}")
                    assert_type_value(obj=value, type_or_value=[str, dict, list], name=f"legend description in {name}")
        if 'model' in response:
            assert_type_value(obj=response['model'], type_or_value=str, name="key 'model' in API response")
        if 'usage' in response:
            assert_type_value(obj=response['usage'], type_or_value=dict, name="key 'usage' in API response")
            for key in ['input_tokens', 'output_tokens']:
                if key in response['usage']:
                    assert_type_value(obj=response['usage'][key], type_or_value=int, name=f"key '{key}' in API response usage")
                    assert_log(expression=response['usage'][key] >= 0, message=f"Expected key '{key}' in API response usage to be non-negative.")
            if 'cost' in response['usage']:
                assert_type_value(obj=response['usage']['cost'], type_or_value=[float, int], name="key 'cost' in API response usage")
                assert_log(expression=math.isfinite(response['usage']['cost']) and response['usage']['cost'] >= 0, message="Expected cost in API response usage to be finite and non-negative.")

    def _format_results(self, state, questions, answers):
        def format_description(value):
            text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
            return text.replace("\n", "\n    ")

        blocks = [f"{escape['yellow']}{escape['bold']}State:{escape['end']}\n    {format_description(state)}"]
        for index, (identifier, question) in enumerate(questions.items()):
            answer = answers[identifier]
            kind = question['type']
            lines = [
                f"{escape['blue']}{escape['bold']}> {index + 1}. {identifier} ({kind}){escape['end']}",
                f"  {escape['cyan']}Question:{escape['end']} {format_description(question['instructions'])}"
            ]
            criteria = question.get('criteria', {})
            probabilities = answer.get('probabilities', {})
            if kind == "noul":
                criteria = {'true': criteria.get('true'), 'false': criteria.get('false')}
                probabilities = {'true': answer['noul'], 'false': 1 - answer['noul']}
            elif kind == "score":
                criteria = {str(i): value for i, value in enumerate(criteria)}
            if criteria:
                lines.append("  Options:")
                for label, description in criteria.items():
                    probability = probabilities.get(label)
                    if kind == "choice":
                        selected = label == answer['choice']
                    elif kind == "noul":
                        selected = probability > 0.5
                    else:
                        selected = probability is not None and probability == max(probabilities.values())
                    marker = "*" if selected else "-"
                    probability_str = "" if probability is None else f" ({probability:.1%})"
                    description_str = "" if description is None else f" {format_description(description)}"
                    line = f"    {marker} {label}{probability_str}{':' if description is not None else ''}"
                    if selected:
                        line = f"{escape['green']}{line}{escape['end']}"
                    lines.append(f"{line}{description_str}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def evaluate(self, state, questions):
        stamp = time.perf_counter()

        # parse arguments
        assert_type_value(obj=state, type_or_value=[str, dict, list], name="argument 'state'")
        self._validate_questions(questions)

        # retrieve API key
        success, message, api_key = self.get_api_key()
        if not success:
            raise UnrecoverableError(message)

        # construct payload
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {api_key}",
            'HTTP-Referer': "https://github.com/ptzld/nimbro-api",
            'X-Title': "NimbRo API"
        }
        if api_key == "":
            del headers['Authorization']
        data = {'state': state, 'questions': questions}
        if self._settings['model'] is not None:
            data['model'] = self._settings['model']

        # use API
        success, message, response = post_request(
            api_name="System One API",
            api_url=self._endpoint['api_url'],
            headers=headers,
            data=data,
            timeout=(self._settings['timeout_connect'], self._settings['timeout_read']),
            logger=self._logger
        )
        if success:
            # parse API response
            try:
                response = response.json()
                self._validate_response(response, questions)
            except Exception as e:
                success = False
                message = f"Failed to parse response from System One API '{self._endpoint['api_url']}': {repr(e)}"

        # finalize response
        if success:
            message = f"Evaluated '{len(questions)}' question{'' if len(questions) == 1 else 's'} in '{time.perf_counter() - stamp:.3f}s'."
            if self._settings['message_results']:
                message = f"{message[:-1]}:\n{self._format_results(state, questions, response['answers'])}"
        else:
            response = None
        return success, message, response
