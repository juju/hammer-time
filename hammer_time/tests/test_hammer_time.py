import asyncio
from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import (
    call,
    patch,
    )

from jujupy.fake import fake_juju_client

from hammer_time.hammer_time import (
    ActionFailed,
    cli_add_remove_many_machine,
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
