import asyncio
from argparse import ArgumentParser
from contextlib import contextmanager
import logging

from juju.application import Application
from juju.client.connection import (
    get_macaroons,
    JujuData,
    )
from jujupy.client import ConditionList
from juju.model import Model
from jujupy import (
    client_for_existing,
    )
from matrix import model
from matrix.tasks.glitch.actions import action
from matrix.tasks.glitch.plan import generate_plan
from matrix.tasks.glitch.main import perform_action
import yaml


class ActionFailed(Exception):
    """Raised when an action failed."""


def get_auth_data(model_client):
    """Get authentication data from a jujupy.ModelClient."""
    jujudata = JujuData()
    jujudata.path = model_client.env.juju_home
    controller_name = model_client.env.controller.name
    cacert = jujudata.controllers()[controller_name].get('ca-cert')
    accounts = jujudata.accounts()[controller_name]
    username = accounts['user']
    password = accounts.get('password')
    return cacert, username, password


def cli_add_remove_many_machine(client):
    """Add and removie many machines using the cli."""
    old_status = client.get_status()
    client.juju('add-machine', ('-n', '5'))
    client.wait_for_started()
    new_status = client.get_status()
    conditions = []
    for machine_id, data in new_status.iter_new_machines(old_status):
        client.juju('remove-machine', (machine_id,))
        conditions.append(client.make_remove_machine_condition(machine_id))
    client.wait_for(ConditionList(conditions))


def add_cli_actions(client):
    """Add cli-based actions.

    This works because @action registers the callable with the Actions
    singleton.
    """
    @action
    async def add_remove_many_machine(
            rule: model.Rule, model: Model, application: Application):
        # Note: application is supplied only to make generate_plan /
        # perform_action happy.  It is ignored.
        cli_add_remove_many_machine(client)


@contextmanager
def connected_model(loop, model_client):
    """Use a jujupy.ModelClient to get a libjuju.Model."""
    host, port = model_client.get_controller_endpoint()
    if ':' in host:
        host = host.join('[]')
    endpoint = '{}:{}'.format(host, port)
    cacert, username, password = get_auth_data(model_client)
    macaroons = get_macaroons() if not password else None
    model = Model(loop)
    loop.run_until_complete(model.connect(
        endpoint, model_client.get_model_uuid(), username, password, cacert,
        macaroons,
        ))
    try:
        yield model
    finally:
        loop.run_until_complete(model.disconnect())


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


def random_plan(plan_file, juju_data, action_count):
    """Implement 'random-plan' subcommand.

    This writes a randomly-generated plan file.
    :param plan_file: The filename for the plan.
    :param juju_data: The JUJU_DATA directory containing the model.
    :param action_count: The number of Glitch actions the plan should include.
    """
    client = client_for_existing(None, juju_data)
    add_cli_actions(client)
    loop = asyncio.get_event_loop()
    with connected_model(loop, client) as model:
        while True:
            plan = loop.run_until_complete(
                generate_plan(None, model, action_count))
            if is_workable_plan(client, plan):
                break
            logging.info('Generated unworkable plan.  Trying again.')
    loop.close()
    with open(plan_file, 'w') as f:
        yaml.safe_dump(plan, f)


def run_glitch(plan, client):
    """Run a Gitch plan against a Juju client.

    :param plan: The parsed glitch plan.
    :param client: The jujupy.ModelClient to run the plan against.
    """
    # Rule is a mandatory, statically-typed argument to perform_action and its
    # callees.
    add_cli_actions(client)
    rule = model.Rule(model.Task(command='glitch', args={'path': None}))
    loop = asyncio.get_event_loop()
    try:
        with connected_model(loop, client) as juju_model:
            for plan_action in plan['actions']:
                logging.info('Performing action {}'.format(plan_action))
                fname, failed = loop.run_until_complete(
                    perform_action(plan_action, juju_model, rule))
                if failed:
                    raise ActionFailed()
    finally:
        loop.close()


def execute_plan(plan_file, juju_data):
    """Implement the 'execute' subcommand.

    :param plan_file: The filename of the plan file to execute.
    :param juju_data: Optional JUJU_DATA for a model to operate on.
    """
    with open(plan_file) as f:
        plan = yaml.safe_load(f)
    client = client_for_existing(None, juju_data)
    client._backend._full_path = client._backend._full_path.decode('utf-8')
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
