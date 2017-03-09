from collections import OrderedDict
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
    AddRemoveManyContainerAction,
    AddRemoveManyMachineAction,
    choose_machine,
    InvalidActionError,
    KillMongoDAction,
    NoValidActionsError,
    random_plan,
    RebootMachineAction,
    run_plan,
    )


def backend_call(client, cmd, args, model=None, check=True, timeout=None,
                 extra_env=None):
    """Return the mock.call for this command."""
    return call(cmd, args, client.used_feature_flags,
                client.env.juju_home, client._cmd_model(True, False), check,
                timeout, extra_env, suppress_err=False)


class TestChooseMachine(TestCase):

    def test_choose_machine(self):
        chosen = set()
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ('-n', '2'))
        for x in range(50):
            chosen.add(choose_machine(client))
            if chosen == {'0', '1'}:
                break
        else:
            raise AssertionError('Did not choose each machine.')

    def test_no_machines(self):
        client = fake_juju_client()
        client.bootstrap()
        with self.assertRaises(InvalidActionError):
            choose_machine(client)


class TestAddRemoveManyMachineAction(TestCase):

    def test_add_remove_many_machine(self):
        client = fake_juju_client()
        client.bootstrap()
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            AddRemoveManyMachineAction.perform(client)
        self.assertEqual([
            backend_call(client, 'add-machine', ('-n', '5')),
            backend_call(client, 'remove-machine', ('0',)),
            backend_call(client, 'remove-machine', ('1',)),
            backend_call(client, 'remove-machine', ('2',)),
            backend_call(client, 'remove-machine', ('3',)),
            backend_call(client, 'remove-machine', ('4',)),
            ], juju_mock.mock_calls)


class TestAddRemoveManyContainerAction(TestCase):

    def test_add_remove_many_container(self):
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ())
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            AddRemoveManyContainerAction.perform(client, '0')
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


class TestKillMongoDAction(TestCase):

    def test_generate_parameters(self):
        client = fake_juju_client()
        client.bootstrap()
        parameters = KillMongoDAction.generate_parameters(client)
        self.assertEqual(parameters, {'machine_id': '0'})
        controller_client = client.get_controller_client()
        controller_client.juju('add-machine', ())
        controller_client.remove_machine('0')
        parameters = KillMongoDAction.generate_parameters(client)
        self.assertEqual(parameters, {'machine_id': '1'})

    def test_perform(self):
        client = fake_juju_client()
        client.bootstrap()
        ctrl_client = client.get_controller_client()
        with patch.object(ctrl_client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            with patch.object(client, 'get_controller_client',
                              return_value=ctrl_client):
                with patch('time.sleep'):
                    KillMongoDAction.perform(client, '0')
        self.assertEqual([
            backend_call(
                ctrl_client, 'ssh',
                ('0',) + KillMongoDAction.kill_script
                ),
            ], juju_mock.mock_calls)


class TestRebootMachineAction(TestCase):

    def test_generate_parameters(self):
        client = fake_juju_client()
        client.bootstrap()
        with self.assertRaises(InvalidActionError):
            parameters = RebootMachineAction.generate_parameters(client)

        client.juju('add-machine', ('-n', '2'))
        client.remove_machine('0')
        parameters = RebootMachineAction.generate_parameters(client)
        self.assertEqual(parameters, {'machine_id': '1'})

    def test_perform(self):
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ())
        parameters = RebootMachineAction.generate_parameters(client)
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            with patch.object(client._backend, 'get_juju_output',
                              autospec=True,
                              side_effect=['earlier', 'now']) as jo_mock:
                RebootMachineAction.perform(client, **parameters)
        self.assertEqual([
            backend_call(client, 'ssh', ('0', 'sudo', 'reboot'), check=False),
            ], juju_mock.mock_calls)
        expected_call = call(
            'ssh', ('0', 'uptime', '-s'),
            client.used_feature_flags, client.env.juju_home,
            client._cmd_model(True, False),
            user_name='admin')
        self.assertEqual([expected_call, expected_call], jo_mock.mock_calls)


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


class Step():

    def __init__(self, test_case, client):
        self.performed = False
        self.test_case = test_case
        self.client = client

    def perform(self, client, bar):
        self.test_case.assertEqual(bar, 'baz')
        self.test_case.assertIs(client, self.client)
        self.performed = True


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

    def test_perform_step(self):

        cur_client = object()

        step = Step(self, cur_client)
        actions = Actions({'step': step})
        actions.perform_step(cur_client, {'step': {'bar': 'baz'}})
        self.assertIs(True, step.performed)


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


class TestRunPlan(TestCase):

    def test_run_plan(self):
        client = fake_juju_client
        step = Step(self, client)
        actions = Actions({'step': step})
        plan = [{'step': {'bar': 'baz'}}]
        with patch('hammer_time.hammer_time.default_actions',
                   autospec=True, return_value=actions):
            run_plan(plan, client)
        self.assertIs(True, step.performed)
