from argparse import ArgumentParser
from contextlib import contextmanager
from random import (
    choice,
    shuffle,
    )
import logging
import re
import subprocess
import sys

from jujupy.client import (
    BaseCondition,
    ConditionList,
    MachineDown,
    )
from jujupy import (
    client_for_existing,
    get_juju_data,
    )
from jujupy.utility import until_timeout
import yaml


def remove_and_wait(client, machines):
    conditions = []
    for machine_id, data in machines:
        conditions.append(client.remove_machine(machine_id))
    client.wait_for(ConditionList(conditions))


class AddRemoveManyMachineAction:

    def generate_parameters(client, status):
        return {}

    def perform(client):
        """Add and removie many machines using the cli."""
        old_status = client.get_status()
        client.juju('add-machine', ('-n', '5'))
        client.wait_for_started()
        new_status = client.get_status()
        remove_and_wait(client, new_status.iter_new_machines(old_status))


class MachineAction:
    """Base class for actions that operate on machines."""

    skip_windows = False

    @classmethod
    def machine_suitable(cls, machine_data):
        return bool(
            not cls.skip_windows or not
            machine_data['series'].startswith('win'))

    @classmethod
    def choose_machine(cls, status):
        machines = list(status.iter_machines(containers=False))
        machines = [(m, d) for m, d in machines
                    if cls.machine_suitable(d)]
        if len(machines) == 0:
            raise InvalidActionError('No suitable machines.')
        return choice(machines)

    @classmethod
    def generate_parameters(cls, client, status):
        machine_id = cls.choose_machine(status)[0]
        return {'machine_id': machine_id}


class RebootMachineAction(MachineAction):
    """Action that reboots a machine."""

    skip_windows = True

    def get_up_since(client, machine_id):
        """Return the date the machine has been up since."""
        return client.get_juju_output('run', '--machine', machine_id,
                                      'uptime -s')

    @classmethod
    def perform(cls, client, machine_id):
        """Add and remove many containers using the cli."""
        up_since = cls.get_up_since(client, machine_id)
        client.juju('run', ('--machine', machine_id, 'sudo', 'reboot'),
                    check=False)
        for x in until_timeout(300):
            try:
                reboot_up_since = cls.get_up_since(client, machine_id)
            except subprocess.CalledProcessError:
                pass
            else:
                if up_since != reboot_up_since:
                    break
        else:
            raise AssertionError('Unable to retrieve uptime.')


def parse_hardware(machine_data):
    hardware = {}
    hardware_str = machine_data.get('hardware')
    if hardware_str is None:
        return None
    for item in hardware_str.split(' '):
        key, value = item.split('=', 1)
        hardware[key] = value
    return hardware


class AddRemoveManyContainerAction(MachineAction):
    """Action to add many containers, then remove them."""

    skip_windows = True

    space_per_instance = 2048

    @classmethod
    def generate_parameters(cls, client, status):
        if client.env.provider == 'lxd':
            raise InvalidActionError('Not supported on LXD provider.')
        machine_id, data = cls.choose_machine(status)
        hardware = parse_hardware(data)
        container_count = cls.calculate_containers(hardware)
        return {
            'container_count': container_count,
            'machine_id': machine_id,
            }

    @classmethod
    def calculate_containers(cls, hardware):
        root_space = int(re.match('^(\d+)M$', hardware['root-disk']).group(1))
        # Allocate only as many containers as will fit on the host.
        # Ensure each instance has cls.space_per_instance.  Subtract 1 for the
        # host.
        # Due to kernel limitations, restrict to 10 containers.  (Actual limit
        # is ~13).
        return min((root_space // cls.space_per_instance) - 1, 10)

    @classmethod
    def machine_suitable(cls, machine_data):
        if not MachineAction.machine_suitable(machine_data):
            return False
        hardware = parse_hardware(machine_data)
        if hardware is None:
            return False
        container_count = cls.calculate_containers(hardware)
        if container_count < 1:
            return False
        return True

    def perform(client, machine_id, container_count):
        """Add and remove many containers using the cli."""
        old_status = client.get_status()
        for count in range(container_count):
            client.juju('add-machine', ('lxd:{}'.format(machine_id)))
        client.wait_for_started()
        new_status = client.get_status()
        new_cont = list(new_status.iter_new_machines(old_status,
                                                     containers=True))
        remove_and_wait(client, sorted(new_cont))


class RunAvailable(BaseCondition):
    """Indicates whether the run operation is available.

    This is a good indicator that the machine is up and running properly.
    """

    def __init__(self, client, machine_id):
        super().__init__()
        self.client = client
        self.machine_id = machine_id

    def iter_blocking_state(self, status):
        exit_status = self.client.juju('run', (
            '--machine', self.machine_id, 'exit 0', '--timeout', '20s'
            ), check=False)
        if exit_status != 0:
            yield (self.machine_id, 'cannot-run')

    def do_raise(self, model_name, status):
        raise Exception(
            'Machine {} cannot run commands.'.format(self.machine_id))


class KillJujuDAction(MachineAction):
    """Action to kill jujud."""

    skip_windows = True

    kill_script = (
        'set -eu;',
        'pid=$(pgrep jujud);'
        'sudo kill $pid;',
        'echo -n Waiting for jujud to die;'
        'while (ps $pid > /dev/null);', 'do',
        '  echo -n .;'
        '  sleep 1;',
        'done;',
        'echo',
        )

    @classmethod
    def perform(cls, client, machine_id):
        # Ideally, we'd use 'run' since it queues until the machine is
        # available.  We can't, because this operation breaks jujud halfway
        # through.  So instead, we wait until the 'run' operation is
        # available, but then use 'ssh'.
        client.wait_for(RunAvailable(client, machine_id))
        client.juju('ssh', (machine_id,) + cls.kill_script)


class KillMongoDAction(MachineAction):
    """Action to kill mongod.  This must operate on controller machine."""

    kill_script = (
        'sudo pkill mongod;',
        'echo -n Waiting for Mongodb to die;'
        'while (pgrep mongod > /dev/null);', 'do',
        '  echo -n .;'
        '  sleep 1;',
        'done;',
        'echo',
        )

    @classmethod
    def generate_parameters(cls, client, status):
        ctrl_client = client.get_controller_client()
        return {'machine_id': cls.choose_machine(ctrl_client.get_status())[0]}

    @classmethod
    def perform(cls, client, machine_id):
        ctrl_client = client.get_controller_client()
        # Ideally, we'd use 'run' since it queues until the machine is
        # available.  We can't, because this operation breaks jujud halfway
        # through.  So instead, we wait until the 'run' operation is
        # available, but then use 'ssh'.
        ctrl_client.wait_for(RunAvailable(client, machine_id))
        ctrl_client.juju('ssh', (machine_id,) + cls.kill_script)


class InterruptNetworkAction(MachineAction):

    skip_windows = True

    def get_command():
        deny_all = '; '.join([
            'iptables --flush',
            'iptables -P FORWARD DROP',
            'iptables -P OUTPUT DROP',
            'iptables -P INPUT DROP',
            ])
        restore = 'iptables-restore < $HOME/iptables'
        commands = [
            'set -eux',
            'sudo iptables-save > $HOME/iptables',
            'echo "{}"| sudo at now + 6 minutes'.format(restore),
            'echo "{}"| sudo at now + 1 minutes'.format(deny_all),
            ]
        return '; '.join(commands)

    @classmethod
    def perform(cls, client, machine_id):
        logging.info('Running: {}'.format(cls.get_command()))
        # Ensure we are not *already* down from some previous operation.
        client.wait_for_started()
        client.juju('run', ('--machine', machine_id, cls.get_command()))
        logging.info(
            ('Waiting for juju to notice that {} is down, so that health'
             ' check does not pass prematurely.').format(machine_id))
        client.wait_for(MachineDown(machine_id))


class AddUnitAction:
    """Add a unit to a random application."""

    def generate_parameters(client, status):
        """Select a random application to add a unit to."""
        status = client.get_status()
        applications = list(status.get_applications())
        if len(applications) == 0:
            raise InvalidActionError('No applications to choose from.')
        return {'application': choice(applications)}

    def perform(client, application):
        """Add a unit to an application."""
        client.juju('add-unit', application)


class RemoveUnitAction:
    """Remove a random unit."""

    def generate_parameters(client, status):
        """Select a random application to add a unit to."""
        status = client.get_status()
        units = list(u for u, d in status.iter_units())
        if len(units) == 0:
            raise InvalidActionError('No units to choose from.')
        return {'unit': choice(units)}

    def perform(client, unit):
        """Add a unit to an application."""
        status = client.get_status()
        # It would be nice to use Status.get_unit, but it does not yet support
        # Python 3.
        for i_unit, data in status.iter_units():
            if i_unit == unit:
                unit_machine = data['machine']
                break
        else:
            raise LookupError(unit)
        client.juju('remove-unit', (unit,))
        return client.make_remove_machine_condition(unit_machine)


def parse_args(argv=None):
    """Parse the arguments of this script."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')
    subparsers.required = True
    rr_parser = subparsers.add_parser(
        'run-random', description='Run random actions and record as a plan.'
        )
    rr_parser.set_defaults(func=run_random)
    rr_parser.add_argument('plan_file', help='The file to write to.')
    rr_parser.add_argument(
        '--force-action',
        help='Force the plan to use this action.',
        choices={
            k for k, v in
            default_actions().list_arbitrary_actions()
            }
        )
    rr_parser.add_argument(
        '--action-count',
        help='Number of actions in the rr_parser.  (default: 1)',
        default=1, type=int)
    rr_parser.add_argument(
        '--unsafe', help='Allow unsafe actions', action='store_true')
    replay_parser = subparsers.add_parser(
        'replay', description='Replay a plan from a previous run.')
    replay_parser.set_defaults(func=replay)
    replay_parser.add_argument('plan_file',
                               help='The file containing the plan.')
    for cur_parser in [rr_parser, replay_parser]:
        cur_parser.add_argument('--juju-data', help='Location of JUJU_DATA.')
        cur_parser.add_argument('--juju-bin', help='Location of juju binary.')
    return parser.parse_args(argv)


class InvalidActionError(Exception):
    """Raised when the action is not valid for the client's model."""


class NoValidActionsError(Exception):
    """Raised when there are no valid actions for the client's model."""


class Actions:

    def __init__(self, initial_actions=None):
        if initial_actions is None:
            initial_actions = {}
        self._actions = initial_actions

    def list_arbitrary_actions(self):
        """Iterate through all known actions in an arbitrary order."""
        action_items = list(self._actions.items())
        shuffle(action_items)
        return action_items

    def generate_step(self, client):
        """Generate an arbitrary action with parameters."""
        status = client.get_status()
        for name, cur_action in self.list_arbitrary_actions():
            try:
                return name, cur_action, cur_action.generate_parameters(client,
                                                                        status)
            except InvalidActionError:
                pass
        else:
            raise NoValidActionsError('No valid actions for model.')

    def perform_step(self, client, step):
        """Perform an action formatted as a step dictionary."""
        ((name, parameters),) = step.items()
        param_str = ', '.join('{}={}'.format(k, repr(v)) for k, v in
                              sorted(parameters.items()))
        logging.info('Performing step: {}({})'.format(name, param_str))
        return self._actions[name].perform(client, **parameters)

    def do_step(self, client, step):
        """Run a step against a ModelClient.

        :param client: The jujupy.ModelClient to run the step against.
        :param step: The step, as a dict.
        """
        condition = self.perform_step(client, step)
        if condition is not None:
            client.wait_for(condition)


def default_actions(unsafe=False):
    action_dict = {
        'add_remove_many_machines': AddRemoveManyMachineAction,
        'add_remove_many_container': AddRemoveManyContainerAction,
        'add_unit': AddUnitAction,
        'interrupt_network': InterruptNetworkAction,
        'kill_jujud': KillJujuDAction,
        'reboot_machine': RebootMachineAction,
        'remove_unit': RemoveUnitAction,
        }
    if unsafe:
        action_dict.update({
            'kill_mongod': KillMongoDAction,
            })
    return Actions(action_dict)


def run_random(plan_file, juju_data, juju_bin, action_count, force_action,
               unsafe):
    """Implement 'random-plan' subcommand.

    This writes a randomly-generated plan file.
    :param plan_file: The filename for the plan.
    :param juju_data: The JUJU_DATA directory containing the model.
    :param juju_bin: Optional path of a juju binary to use.
    :param action_count: The number of actions the plan should include.
    :param force_action: If non-None, the action to use for generating the
        plan.
    :param unsafe: If True, include known-unsafe operations in plans.  Has no
        effect when force_action is supplied.
    """
    with checked_client(juju_bin, juju_data) as client:
        plan = []
        if force_action is not None:
            actions = default_actions(unsafe=True)
            actions = Actions({force_action: actions._actions[force_action]})
        else:
            actions = default_actions(unsafe)
        for step_num in range(action_count):
            try:
                name, action, parameters = actions.generate_step(client)
            except NoValidActionsError as e:
                print(e, file=sys.stderr)
                sys.exit(1)
            step = {name: parameters}
            plan.append(step)
            with open(plan_file, 'w') as f:
                yaml.safe_dump(plan, f)
            actions.do_step(client, step)


def run_plan(plan, client):
    """Run a plan against a ModelClient.

    :param plan: The plan, as a list of dicts.
    :param client: The jujupy.ModelClient to run the plan against.
    """
    actions = default_actions()
    for step in plan:
        actions.do_step(client, step)


@contextmanager
def checked_client(juju_bin, juju_data):
    if juju_data is None:
        juju_data = get_juju_data()
    client = client_for_existing(juju_bin, juju_data)
    # Ensure the model is healthy before beginning.
    client.wait_for_started()
    yield client
    # Ensure the model is healthy after running the plan.
    client.wait_for_started()


def replay(plan_file, juju_data, juju_bin):
    """Implement the 'replay' subcommand.

    :param plan_file: The filename of the plan file to replay.
    :param juju_data: Optional JUJU_DATA for a model to operate on.
    :param juju_bin: Optional path of a juju binary to use.
    """
    with open(plan_file) as f:
        plan = yaml.safe_load(f)
    with checked_client(juju_bin, juju_data) as client:
        run_plan(plan, client)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    # convert the parsed args to kwargs, dropping 'func' and 'cmd', since they
    # are not useful as kwargs.
    kwargs = dict((k, v) for k, v in vars(args).items()
                  if k not in ('func', 'cmd'))
    args.func(**kwargs)
