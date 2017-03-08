from argparse import ArgumentParser
from random import (
    choice,
    shuffle,
    )
import logging

from jujupy.client import ConditionList
from jujupy import (
    client_for_existing,
    )
import yaml


def remove_and_wait(client, machines):
    conditions = []
    for machine_id, data in machines:
        conditions.append(client.remove_machine(machine_id))
    client.wait_for(ConditionList(conditions))


def cli_add_remove_many_machine(client):
    """Add and removie many machines using the cli."""
    old_status = client.get_status()
    client.juju('add-machine', ('-n', '5'))
    client.wait_for_started()
    new_status = client.get_status()
    remove_and_wait(client, new_status.iter_new_machines(old_status))


class AddRemoveManyMachineAction:

    def generate_parameters(client):
        return {}

    perform = cli_add_remove_many_machine


def cli_add_remove_many_container(client, host_id):
    """Add and remove many containers using the cli."""
    old_status = client.get_status()
    for count in range(8):
        client.juju('add-machine', ('lxd:{}'.format(host_id)))
    client.wait_for_started()
    new_status = client.get_status()
    new_cont = list(new_status.iter_new_machines(old_status,
                                                 containers=True))
    remove_and_wait(client, sorted(new_cont))


class AddRemoveManyContainerAction:

    def generate_parameters(client):
        status = client.get_status()
        machines = list(m for m, d in status.iter_machines(containers=False))
        return {'host_id': choice(machines)}

    perform = cli_add_remove_many_container


def parse_args():
    """Parse the arguments of this script."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')
    subparsers.required = True
    plan = subparsers.add_parser(
        'random-plan', description='Generate a random plan.'
        )
    plan.set_defaults(func=random_plan)
    plan.add_argument('plan_file', help='The file to write to.')
    plan.add_argument('--juju-data', help='Location of JUJU_DATA.')
    plan.add_argument(
        '--action-count', help='Number of actions in the plan.  (default: 1)',
        default=1, type=int)

    execute = subparsers.add_parser(
        'execute', description='Execute a plan.')
    execute.set_defaults(func=execute_plan)
    execute.add_argument('plan_file', help='The file containing the plan.')
    execute.add_argument('--juju-data', help='Location of JUJU_DATA.')
    return parser.parse_args()


def is_workable_plan(client, plan):
    """Check whether the supplied plan is workable for the current model.

    Currently, this just checks whether the plan wants to remove the last unit
    of an application.  More checks may be added in the future.
    """
    for plan_action in plan['actions']:
        if plan_action['action'] == 'remove_unit':
            status = client.get_status()
            for selector in plan_action['selectors']:
                if selector['selector'] == 'units':
                    applications = status.get_applications()
                    units = applications[selector['application']]['units']
                    if len(units) < 2:
                        return False
    return True


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
        for name, cur_action in self.list_arbitrary_actions():
            try:
                return name, cur_action, cur_action.generate_parameters(client)
            except InvalidActionError:
                pass
        else:
            raise NoValidActionsError('No valid actions for model.')

    def perform_step(self, client, step):
        """Perform an action formatted as a step dictionary."""
        ((name, parameters),) = step.items()
        self._actions[name].perform(client, **parameters)


def default_actions():
    return Actions({
        'add_remove_many_machines': AddRemoveManyMachineAction,
        'add_remove_many_container': AddRemoveManyContainerAction,
        })


def random_plan(plan_file, juju_data, action_count):
    """Implement 'random-plan' subcommand.

    This writes a randomly-generated plan file.
    :param plan_file: The filename for the plan.
    :param juju_data: The JUJU_DATA directory containing the model.
    :param action_count: The number of actions the plan should include.
    """
    client = client_for_existing(None, juju_data)
    plan = []
    actions = default_actions()
    for step in range(action_count):
        name, action, parameters = actions.generate_step(client)
        plan.append({name: parameters})
    with open(plan_file, 'w') as f:
        yaml.dump(plan, f)


def run_glitch(plan, client):
    """Run a plan against a ModelClient.

    :param plan: The plan, as a list of dicts.
    :param client: The jujupy.ModelClient to run the plan agains.
    """
    actions = default_actions()
    for step in plan:
        actions.perform_step(client, step)


def execute_plan(plan_file, juju_data):
    """Implement the 'execute' subcommand.

    :param plan_file: The filename of the plan file to execute.
    :param juju_data: Optional JUJU_DATA for a model to operate on.
    """
    with open(plan_file) as f:
        plan = yaml.safe_load(f)
    client = client_for_existing(None, juju_data)
    # Ensure the model is healthy before beginning.
    client.wait_for_started()
    run_glitch(plan, client)
    # Ensure the model is healthy after running the plan.
    client.wait_for_started()


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    # convert the parsed args to kwargs, dropping 'func' and 'cmd', since they
    # are not useful as kwargs.
    kwargs = dict((k, v) for k, v in vars(args).items()
                  if k not in ('func', 'cmd'))
    args.func(**kwargs)
