from argparse import Namespace
from collections import OrderedDict
from contextlib import contextmanager
import os
from unittest import TestCase
from unittest.mock import (
    call,
    patch,
    )

from jujupy import Status
from jujupy.client import ProvisioningError
from jujupy.fake import (
    fake_juju_client,
    )

from jujupy.utility import temp_dir
import yaml

from hammer_time import hammer_time as ht
from hammer_time.hammer_time import (
    Actions,
    AddRemoveManyContainerAction,
    AddRemoveManyMachineAction,
    AddUnitAction,
    checked_client,
    choose_machine,
    default_actions,
    InterruptNetworkAction,
    InvalidActionError,
    KillJujuDAction,
    KillMongoDAction,
    MachineAction,
    NoValidActionsError,
    parse_args,
    replay,
    RunAvailable,
    run_random,
    RebootMachineAction,
    RemoveUnitAction,
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
        status = client.get_status()
        for x in range(50):
            chosen.add(choose_machine(status))
            if chosen == {'0', '1'}:
                break
        else:
            raise AssertionError('Did not choose each machine.')

    def test_no_machines(self):
        client = fake_juju_client()
        client.bootstrap()
        status = client.get_status()
        with self.assertRaises(InvalidActionError):
            choose_machine(status)

    def test_skip_windows(self):
        status = Status({'machines': {
            '0': {'series': 'winfoo'},
            '1': {'series': 'angsty'},
            }}, '')
        for x in range(50):
            if choose_machine(status, skip_windows=True) == '0':
                raise AssertionError('Chose windows machine.')
        status_2 = Status({'machines': {
            '0': {'series': 'winfoo'},
            }}, '')
        with self.assertRaises(InvalidActionError):
            choose_machine(status_2, skip_windows=True)


class TestMachineAction(TestCase):

    def test_generate_parameters(self):
        client = fake_juju_client()
        client.bootstrap()
        with self.assertRaises(InvalidActionError):
            parameters = MachineAction.generate_parameters(
                client, client.get_status())

        client.juju('add-machine', ('-n', '2'))
        client.remove_machine('0')
        parameters = MachineAction.generate_parameters(
            client, client.get_status())
        self.assertEqual(parameters, {'machine_id': '1'})

    def test_generate_parameters_no_windows(self):

        class MachineActionNoWindows(MachineAction):

            skip_windows = True

        status = Status({'machines': {
            '1': {'series': 'winfoo'},
            }}, '')
        with self.assertRaises(InvalidActionError):
            parameters = MachineActionNoWindows.generate_parameters(
                None, status)
        status_2 = Status({'machines': {
            '1': {'series': 'wifoo'},
            }}, '')
        parameters = MachineActionNoWindows.generate_parameters(
            None, status_2)
        self.assertEqual(parameters, {'machine_id': '1'})


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
            perform(AddRemoveManyContainerAction, client)
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


def perform(cls, client):
    """Generate parameters and then perform with them."""
    parameters = cls.generate_parameters(client, client.get_status())
    return cls.perform(client, **parameters)


class TestRunAvailable(TestCase):

    def test_iter_blocking_state(self):
        client = fake_juju_client()
        ra = RunAvailable(client, '0')
        with patch.object(client._backend, 'juju',
                          return_value=1) as juju_mock:
            self.assertEqual([('0', 'cannot-run')],
                             list(ra.iter_blocking_state(None)))
            juju_mock.return_value = 0
            self.assertEqual([],
                             list(ra.iter_blocking_state(None)))
        self.assertEqual([backend_call(
            client, 'run',
            ('--machine', '0', 'exit 0', '--timeout', '20s'), check=False
            )] * 2,
            juju_mock.mock_calls)

    def test_do_raise(self):
        ra = RunAvailable(None, 'asdf')
        with self.assertRaisesRegex(Exception,
                                    'Machine asdf cannot run commands.'):
            ra.do_raise(None, None)


class TestKillJujuDAction(TestCase):

    def test_perform(self):
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ())
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            with patch.object(RunAvailable, 'iter_blocking_state',
                              return_value=iter([])):
                perform(KillJujuDAction, client)
        self.assertEqual([
            backend_call(
                client, 'ssh',
                ('0',) + KillJujuDAction.kill_script
                ),
            ], juju_mock.mock_calls)


class TestKillMongoDAction(TestCase):

    def test_generate_parameters(self):
        client = fake_juju_client()
        client.bootstrap()
        parameters = KillMongoDAction.generate_parameters(
            client, client.get_status())
        self.assertEqual(parameters, {'machine_id': '0'})
        controller_client = client.get_controller_client()
        controller_client.juju('add-machine', ())
        controller_client.remove_machine('0')
        parameters = KillMongoDAction.generate_parameters(
            client, client.get_status())
        self.assertEqual(parameters, {'machine_id': '1'})

    def test_perform(self):
        client = fake_juju_client()
        client.bootstrap()
        ctrl_client = client.get_controller_client()
        with patch.object(ctrl_client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            with patch.object(client, 'get_controller_client',
                              return_value=ctrl_client):
                with patch.object(RunAvailable, 'iter_blocking_state',
                                  return_value=iter([])):
                            perform(KillMongoDAction, client)
            self.assertEqual([
                backend_call(
                    ctrl_client, 'ssh',
                    ('0',) + KillMongoDAction.kill_script
                    ),
                ], juju_mock.mock_calls)


class TestInterruptNetworkAction(TestCase):

    def test_perform(self):
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ())
        up_status = Status({'machines': {'0': {
            'juju-status': {'current': 'started'},
            'series': 'angsty',
            }}}, '')
        down_status = Status({'machines': {'0': {
            'juju-status': {'current': 'down'},
            'series': 'angsty',
            }}}, '')
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            with patch.object(client, 'get_status', side_effect=[
                              up_status, up_status, down_status]):
                perform(InterruptNetworkAction, client)
        self.assertEqual([
            backend_call(
                client, 'run',
                ('--machine', '0', InterruptNetworkAction.get_command())
                ),
            ], juju_mock.mock_calls)


class TestRebootMachineAction(TestCase):

    def test_perform(self):
        client = fake_juju_client()
        client.bootstrap()
        client.juju('add-machine', ())
        parameters = RebootMachineAction.generate_parameters(
            client, client.get_status())
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            with patch.object(client._backend, 'get_juju_output',
                              autospec=True,
                              side_effect=['earlier', 'now']) as jo_mock:
                RebootMachineAction.perform(client, **parameters)
        self.assertEqual([
            backend_call(client, 'run', ('--machine', '0', 'sudo', 'reboot'),
                         check=False),
            ], juju_mock.mock_calls)
        expected_call = call(
            'run', ('--machine', '0', 'uptime -s'),
            client.used_feature_flags, client.env.juju_home,
            client._cmd_model(True, False),
            user_name='admin')
        self.assertEqual([expected_call, expected_call], jo_mock.mock_calls)


class TestAddUnitAction(TestCase):

    def test_generate_parameters(self):
        client = fake_juju_client()
        client.bootstrap()
        with self.assertRaisesRegex(InvalidActionError,
                                    'No applications to choose from.'):
            AddUnitAction.generate_parameters(client, client.get_status())
        client.deploy('app1')
        client.deploy('app2')
        parameter_variations = set()
        for count in range(50):
            parameter_variations.add(
                tuple(AddUnitAction.generate_parameters(
                    client, client.get_status()).items()))
            if parameter_variations == {
                        (('application', 'app1'),),
                        (('application', 'app2'),),
                    }:
                break
        else:
            raise AssertionError(
                'One of the expected apps was never selected.')

    def test_perform_action(self):
        client = fake_juju_client()
        client.bootstrap()
        client.deploy('app1')
        parameters = AddUnitAction.generate_parameters(
            client, client.get_status())
        AddUnitAction.perform(client, **parameters)
        status = client.get_status()
        self.assertEqual(
            {'app1/0', 'app1/1'},
            {u for u, d in status.iter_units()},
            )


class TestRemoveUnitAction(TestCase):

    def test_generate_parameters(self):
        client = fake_juju_client()
        client.bootstrap()
        with self.assertRaisesRegex(InvalidActionError,
                                    'No units to choose from.'):
            RemoveUnitAction.generate_parameters(client, client.get_status())
        client.deploy('app1')
        client.deploy('app2')
        parameter_variations = set()
        for count in range(50):
            parameter_variations.add(
                tuple(RemoveUnitAction.generate_parameters(
                    client, client.get_status()).items()))
            if parameter_variations == {
                        (('unit', 'app1/0'),),
                        (('unit', 'app2/0'),),
                    }:
                break
        else:
            raise AssertionError(
                'One of the expected units was never selected.')

    def test_perform_action(self):
        client = fake_juju_client()
        client.bootstrap()
        client.deploy('app1')
        parameters = RemoveUnitAction.generate_parameters(
            client, client.get_status())
        condition = RemoveUnitAction.perform(client, **parameters)
        status = client.get_status()
        self.assertEqual(
            set(), {u for u, d in status.iter_units()})
        expected = client.make_remove_machine_condition('0')
        self.assertEqual(condition, expected)


class TestParseArgs(TestCase):

    def test_run_random_defaults(self):
        args = parse_args(['run-random', 'myplan'])
        self.assertEqual(args, Namespace(
            action_count=1, cmd='run-random', force_action=None,
            func=run_random, juju_data=None, plan_file='myplan',
            unsafe=False, juju_bin=None,
            ))

    def test_run_random_unsafe(self):
        args = parse_args(['run-random', 'myplan', '--unsafe'])
        self.assertIs(True, args.unsafe)

    def test_run_random_juju_bin(self):
        args = parse_args(['run-random', 'myplan', '--juju-bin', 'asdf'])
        self.assertEqual('asdf', args.juju_bin)

    def test_replay_defaults(self):
        args = parse_args(['replay', 'myplan'])
        self.assertEqual(args, Namespace(
            cmd='replay', func=replay, juju_data=None, plan_file='myplan',
            juju_bin=None,
            ))

    def test_replay_juju_bin(self):
        args = parse_args(['replay', 'myplan', '--juju-bin', 'asdf'])
        self.assertEqual('asdf', args.juju_bin)


class FixedOrderActions(Actions):

    def __init__(self, items):
        self._actions = OrderedDict(items)

    def list_arbitrary_actions(self):
        return list(self._actions.items())


class FooBarAction:

    def __init__(self, client, wait_for=False, raise_exception=False):
        self.client = client
        self.performed = False
        self.wait_for = wait_for
        self.raise_exception = raise_exception

    def generate_parameters(self, client, status):
        assert self.client is client
        return {'foo': 'bar'}

    def perform(self, client, foo):
        if self.raise_exception:
            raise Exception()
        self.performed = True
        if self.wait_for:
            return RaiseCondition


class WaitForException(Exception):
    pass


class RaiseCondition:

    already_satisfied = False

    timeout = None

    @staticmethod
    def iter_blocking_state(client):
        raise WaitForException
        yield


class Step():

    def __init__(self, test_case, client, wait_for=False):
        self.performed = False
        self.test_case = test_case
        self.client = client
        self.wait_for = wait_for

    def perform(self, client, bar):
        self.test_case.assertEqual(bar, 'baz')
        self.test_case.assertIs(client, self.client)
        self.performed = True
        if self.wait_for:
            return RaiseCondition


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

        def generate_parameters(client, status):
            raise InvalidActionError()

    def test_generate_step(self):
        cur_client = fake_juju_client()
        cur_client.bootstrap()
        foo_bar = FooBarAction(cur_client)
        actions = FixedOrderActions([('one', foo_bar)])
        self.assertEqual(actions.generate_step(cur_client),
                         ('one', foo_bar, {'foo': 'bar'}))

    def test_generate_step_skip_invalid(self):
        cur_client = fake_juju_client()
        cur_client.bootstrap()
        foo_bar = FooBarAction(cur_client)
        actions = FixedOrderActions([
            ('one', self.InvalidAction), ('two', foo_bar)])
        self.assertEqual(actions.generate_step(cur_client),
                         ('two', foo_bar, {'foo': 'bar'}))

    def test_generate_step_no_valid(self):
        cur_client = fake_juju_client()
        cur_client.bootstrap()
        actions = FixedOrderActions([('one', self.InvalidAction)])
        with self.assertRaisesRegex(
                NoValidActionsError, 'No valid actions for model.'):
            actions.generate_step(cur_client)

    def test_perform_step(self):

        cur_client = object()

        step = Step(self, cur_client)
        actions = Actions({'step': step})
        actions.perform_step(cur_client, {'step': {'bar': 'baz'}})
        self.assertIs(True, step.performed)


@contextmanager
def client_and_plan():
    cur_client = fake_juju_client()
    cur_client.bootstrap()
    with patch('hammer_time.hammer_time.client_for_existing',
               return_value=cur_client, autospec=True) as cfe_mock:
        with temp_dir() as plan_dir:
            plan_file = os.path.join(plan_dir, 'asdf.yaml')
            yield (cur_client, cfe_mock, plan_file)


@contextmanager
def patch_actions(action_list):
    actions = FixedOrderActions(action_list)
    with patch('hammer_time.hammer_time.default_actions',
               autospec=True, return_value=actions):
        yield actions


class TestRunRandom(TestCase):

    def test_run_random(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            foo_bar = FooBarAction(cur_client)
            foo_bar_raise = FooBarAction(cur_client, wait_for=True)
            with patch_actions([('one', foo_bar),
                                ('two', foo_bar_raise)]) as actions:
                with self.assertRaises(WaitForException):
                    with patch.object(
                            actions, 'list_arbitrary_actions',
                            side_effect=[
                                [('one', foo_bar)],
                                [('two', foo_bar_raise)],
                                ]):
                        run_random(plan_file, 'fasd', None, 3, None,
                                   unsafe=False)
            with open(plan_file) as f:
                plan = yaml.load(f)
        self.assertIs(True, foo_bar.performed)
        cfe_mock.assert_called_once_with(None, 'fasd')
        self.assertEqual(plan, [
            {'one': {'foo': 'bar'}}, {'two': {'foo': 'bar'}}])

    def test_run_random_force_action(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            run_random(plan_file, 'fasd', None, 1, 'add_remove_many_machines',
                       unsafe=False)
            with open(plan_file) as f:
                plan = yaml.load(f)
        self.assertEqual(plan, [
            {'add_remove_many_machines': {}}])

    def test_run_random_action_exception(self):
        # Even when a step raises an exception, its plan is still written.
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            foo_bar = FooBarAction(cur_client, raise_exception=True)
            with patch_actions([('one', foo_bar)]):
                with self.assertRaises(Exception):
                    run_random(plan_file, 'fasd', None, 3, None, unsafe=False)
            with open(plan_file) as f:
                plan = yaml.load(f)
        cfe_mock.assert_called_once_with(None, 'fasd')
        self.assertEqual(plan, [
            {'one': {'foo': 'bar'}}])

    def test_run_random_unsafe_true(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            with patch('hammer_time.hammer_time.default_actions',
                       wraps=default_actions) as da_mock:
                with patch.object(RunAvailable, 'iter_blocking_state',
                                  return_value=iter([])):
                    run_random(plan_file, 'fasd', None, 3, None, unsafe=True)
        da_mock.assert_called_once_with(True)

    def test_run_random_unsafe_false(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            with patch('hammer_time.hammer_time.default_actions',
                       wraps=default_actions) as da_mock:
                run_random(plan_file, 'fasd', None, 3, None, unsafe=False)
        da_mock.assert_called_once_with(False)

    def test_run_random_unsafe_force_action(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            with patch('hammer_time.hammer_time.default_actions',
                       wraps=default_actions) as da_mock:
                with patch.object(RunAvailable, 'iter_blocking_state',
                                  return_value=iter([])):
                        run_random(plan_file, 'fasd', None, 3, 'kill_mongod',
                                   unsafe=False)
        da_mock.assert_called_once_with(unsafe=True)

    def test_run_random_juju_bin(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            foo_bar = FooBarAction(cur_client)
            with patch_actions([('one', foo_bar)]):
                run_random(plan_file, 'fasd', 'juju1', 3, None,
                           unsafe=False)
        cfe_mock.assert_called_once_with('juju1', 'fasd')

    def test_run_random_checks_before(self):
        error_status = {'machines': {'0': {'machine-status': {
            'current': 'provisioning error',
            }}}}
        with client_and_plan() as (client, cfe_mock, plan_file):
            foo_bar_raise = FooBarAction(client, raise_exception=True)
            models = client._backend.controller_state.models
            model_state = models[client.model_name]
            with patch.object(model_state, 'get_status_dict',
                              return_value=error_status, autospec=True):
                with self.assertRaises(ProvisioningError):
                    with patch_actions([('one', foo_bar_raise)]):
                        run_random(plan_file, 'fasd', 'juju1', 1, None,
                                   unsafe=False)

    def test_run_random_checks_after(self):
        status = {'machines': {}}

        def perform(foo_bar_self, client, foo):
            status['machines'].update({
                '0': {'machine-status': {'current': 'provisioning error'}}
                })

        with client_and_plan() as (client, cfe_mock, plan_file):
            foo_bar = FooBarAction(client)
            models = client._backend.controller_state.models
            model_state = models[client.model_name]
            with patch.object(model_state, 'get_status_dict',
                              return_value=status, autospec=True):
                with patch.object(FooBarAction, 'perform', perform):
                    with patch_actions([('one', foo_bar)]):
                        with self.assertRaises(ProvisioningError):
                            run_random(plan_file, 'fasd', 'juju1', 1, None,
                                       unsafe=False)


class TestDefaultActions(TestCase):

    def test_unsafe(self):
        actions = default_actions(unsafe=True)
        self.assertIn('kill_mongod', actions._actions)
        actions = default_actions(unsafe=False)
        self.assertNotIn('kill_mongod', actions._actions)


class TestRunPlan(TestCase):

    @contextmanager
    def run_cxt(self, wait_for=False):
        client = fake_juju_client()
        client.bootstrap()
        step = Step(self, client, wait_for=wait_for)
        actions = Actions({'step': step})
        plan = [{'step': {'bar': 'baz'}}]
        with patch('hammer_time.hammer_time.default_actions',
                   autospec=True, return_value=actions):
            yield client, plan, step

    def test_run_plan(self):
        with self.run_cxt() as (client, plan, step):
            run_plan(plan, client)
        self.assertIs(True, step.performed)

    def test_run_plan_wait_for(self):
        with self.run_cxt(wait_for=True) as (client, plan, step):
            with self.assertRaises(WaitForException):
                run_plan(plan, client)


class TestCheckedClient(TestCase):

    def test_juju_data_default(self):
        with patch.object(ht, 'client_for_existing') as cfe_mock:
            with patch.dict(os.environ, {'JUJU_DATA': 'bar'}):
                with checked_client('foo', None) as client:
                    pass
        self.assertIs(client, cfe_mock.return_value)
        cfe_mock.assert_called_once_with('foo', 'bar')

    def test_juju_data_supplied(self):
        with patch.object(ht, 'client_for_existing') as cfe_mock:
            with patch.dict(os.environ, {'JUJU_DATA': 'bar'}):
                with checked_client('foo', 'qux'):
                    pass
        cfe_mock.assert_called_once_with('foo', 'qux')


class TestReplay(TestCase):

    def test_run_random_juju_bin(self):
        with client_and_plan() as (cur_client, cfe_mock, plan_file):
            with open(plan_file, 'w') as f:
                yaml.safe_dump([], f)
            replay(plan_file, 'fasd', 'juju1')
        cfe_mock.assert_called_once_with('juju1', 'fasd')
