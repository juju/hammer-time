import asyncio
from collections import OrderedDict
from contextlib import contextmanager
import os
from unittest import TestCase
from unittest.mock import (
    call,
    patch,
    )

from jujupy.fake import fake_juju_client
from jujupy.utility import temp_dir
import yaml

from hammer_time.hammer_time import (
    Actions,
    ActionFailed,
    cli_add_remove_many_container,
    cli_add_remove_many_machine,
    InvalidActionError,
    NoValidActionsError,
    random_plan,
    run_glitch,
    )


def backend_call(client, cmd, args, model=None, check=True, timeout=None,
                 extra_env=None):
    """Return the mock.call for this command."""
    return call(cmd, args, client.used_feature_flags,
                client.env.juju_home, client._cmd_model(True, False), check,
                timeout, extra_env, suppress_err=False)


class TestCLIAddRemoveManyMachine(TestCase):

    def test_add_remove_many_machine(self):
        client = fake_juju_client()
        client.bootstrap()
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            cli_add_remove_many_machine(client)
        self.assertEqual([
            backend_call(client, 'add-machine', ('-n', '5')),
            backend_call(client, 'remove-machine', ('0',)),
            backend_call(client, 'remove-machine', ('1',)),
            backend_call(client, 'remove-machine', ('2',)),
            backend_call(client, 'remove-machine', ('3',)),
            backend_call(client, 'remove-machine', ('4',)),
            ], juju_mock.mock_calls)


class TestCLIAddRemoveManyContainer(TestCase):

    def test_add_remove_many_container(self):
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ())
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            cli_add_remove_many_container(client, '0')
        self.assertEqual([
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'add-machine', ('lxd:0')),
            backend_call(client, 'remove-machine', ('0/lxd/0',)),
            backend_call(client, 'remove-machine', ('0/lxd/1',)),
            backend_call(client, 'remove-machine', ('0/lxd/2',)),
            backend_call(client, 'remove-machine', ('0/lxd/3',)),
            backend_call(client, 'remove-machine', ('0/lxd/4',)),
            backend_call(client, 'remove-machine', ('0/lxd/5',)),
            backend_call(client, 'remove-machine', ('0/lxd/6',)),
            backend_call(client, 'remove-machine', ('0/lxd/7',)),
            ], juju_mock.mock_calls)


@contextmanager
def plan_fakes():

    async def fail(a, b, c):
        raise Exception()

    async def do_pass(a, b, c):
        pass

    actions = {
        'fail': {'func': fail},
        'pass': {'func': do_pass},
        }

    async def dummy(a, b):

        return [None]

    selectors = {
        'dummy': dummy,
    }
    from matrix.tasks.glitch import main
    with patch.object(main, 'Actions', actions):
        with patch.object(main, 'Selectors', selectors):
            yield


class FixedOrderActions(Actions):

    def __init__(self, items):
        self._actions = OrderedDict(items)

    def list_arbitrary_actions(self):
        return list(self._actions.items())


class FooBarAction:

    def __init__(self, client):
        self.client = client

    def generate_parameters(self, client):
        assert self.client is client
        return {'foo': 'bar'}


class TestActions(TestCase):

    def test_list_arbitrary_actions(self):
        actions = Actions()
        self.assertEqual([], actions.list_arbitrary_actions())
        one_obj = object()
        actions = Actions({'one': one_obj})
        self.assertEqual([('one', one_obj)],
                         actions.list_arbitrary_actions())

    def test_list_arbitrary_actions_is_random(self):
        one_obj = object()
        two_obj = object()
        actions = Actions({'one': one_obj, 'two': two_obj})
        self.assert_sequence(actions, [('one', one_obj), ('two', two_obj)])
        self.assert_sequence(actions, [('two', two_obj), ('one', one_obj)])

    def assert_sequence(self, actions, expected):
        for x in range(50):
            action_list = actions.list_arbitrary_actions()
            if action_list == expected:
                break
        else:
            raise AssertionError(
                'Never got expected sequence.  Got {}'.format(action_list))

    class InvalidAction:

        def generate_parameters(client):
            raise InvalidActionError()

    def test_generate_step(self):
        cur_client = fake_juju_client()
        foo_bar = FooBarAction(cur_client)
        actions = FixedOrderActions([('one', foo_bar)])
        self.assertEqual(actions.generate_step(cur_client),
                         ('one', foo_bar, {'foo': 'bar'}))

    def test_generate_step_skip_invalid(self):
        cur_client = fake_juju_client()
        foo_bar = FooBarAction(cur_client)
        actions = FixedOrderActions([
            ('one', self.InvalidAction), ('two', foo_bar)])
        self.assertEqual(actions.generate_step(cur_client),
                         ('two', foo_bar, {'foo': 'bar'}))

    def test_generate_step_no_valid(self):
        actions = FixedOrderActions([('one', self.InvalidAction)])
        with self.assertRaisesRegex(
                NoValidActionsError, 'No valid actions for model.'):
            actions.generate_step(None)


class TestRandomPlan(TestCase):

    def test_random_plan(self):
        cur_client = fake_juju_client()
        with patch('hammer_time.hammer_time.client_for_existing',
                   return_value=cur_client, autospec=True) as cfe_mock:
            foo_bar = FooBarAction(cur_client)
            actions = FixedOrderActions([('one', foo_bar)])
            with temp_dir() as plan_dir:
                plan_file = os.path.join(plan_dir, 'asdf.yaml')
                with patch('hammer_time.hammer_time.default_actions',
                           autospec=True, return_value=actions):
                    random_plan(plan_file, 'fasd', 2)
                with open(plan_file) as f:
                    plan = yaml.load(f)
        cfe_mock.assert_called_once_with(None, 'fasd')
        self.assertEqual(plan, [
            {'one': {'foo': 'bar'}}, {'one': {'foo': 'bar'}}])


class TestRunGlitch(TestCase):

    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_run_glitch(self):
        plan = {
            'actions': [{
                'action': 'pass', 'selectors': [{'selector': 'dummy'}]
                }]
            }
        with patch('hammer_time.hammer_time.connected_model'):
            with plan_fakes():
                run_glitch(plan, None)
        self.assertIs(True, asyncio.get_event_loop().is_closed())

    def test_run_glitch_raises_action_failed(self):
        plan = {
            'actions': [{
                'action': 'fail', 'selectors': [{'selector': 'dummy'}]
                }]
            }
        with patch('hammer_time.hammer_time.connected_model'):
            with plan_fakes():
                with self.assertRaises(ActionFailed):
                    run_glitch(plan, None)
        self.assertIs(True, asyncio.get_event_loop().is_closed())
