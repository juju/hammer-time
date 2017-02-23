import asyncio
from argparse import ArgumentParser
from contextlib import contextmanager
import logging
import os
import subprocess
import shlex
import sys

from juju.client.connection import (
    Connection,
    get_macaroons,
    JujuData,
    )
from juju.model import Model
from jujupy import (
    get_client_class,
    ModelClient,
    client_for_existing,
    )
from jujupy.client import Controller
import yaml

from matrix.bus import Bus
from matrix import model
from matrix.tasks.glitch.plan import generate_plan
from matrix.tasks.glitch.main import perform_action


class NoCurrentController(Exception):
    """Raised when Juju has no current controller set."""


class NoEndpoints(Exception):
    """Raised when the controller endpoints are not yet set."""


class SpecificJujuData(JujuData):
    """A JujuData for a specific path."""

    def __init__(self, juju_data_path):
        super().__init__()
        if juju_data_path is not None:
            self.path = juju_data_path

    def current_controller(self):
        cmd = shlex.split('juju list-controllers --format yaml')
        env = dict(os.environ)
        env['JUJU_DATA'] = self.path
        output = subprocess.check_output(cmd, env=env)
        output = yaml.safe_load(output)
        return output.get('current-controller', '')


def get_auth_data(model_client):
    jujudata = JujuData()
    jujudata.path = model_client.env.juju_home
    controller_name = model_client.env.controller.name
    cacert = jujudata.controllers()[controller_name].get('ca-cert')
    accounts = jujudata.accounts()[controller_name]
    username = accounts['user']
    password = accounts.get('password')
    return cacert, username, password


@contextmanager
def connected_model(loop, model_client):
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
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')
    subparsers.required = True
    plan = subparsers.add_parser('random-plan')
    plan.set_defaults(func=make_plan)
    plan.add_argument('plan_file', help='The file to write to.')
    plan.add_argument('--juju-data', help='Location of JUJU_DATA.')
    plan.add_argument('--action-count', help='Number of actions in the plan',
                      default=1, type=int)
    execute = subparsers.add_parser('execute')
    execute.set_defaults(func=execute_plan)
    execute.add_argument('plan_file', help='The file containing the plan.')
    execute.add_argument('--juju-data', help='Location of JUJU_DATA.')
    return parser.parse_args()


def is_workable_plan(client, plan):
    """Check whether the current plan is workable.
    """
    for action in plan['actions']:
        if action['action'] == 'remove_unit':
            status = client.get_status()
            for selector in action['selectors']:
                if selector['selector'] == 'units':
                    applications = status.get_applications()
                    units = applications[selector['application']]['units']
                    if len(units) < 2:
                        return False
    return True



def make_plan(plan_file, juju_data, action_count):
    client = client_for_existing(None, juju_data)
    loop = asyncio.get_event_loop()
    with connected_model(loop, client) as model:
        while True:
            plan = loop.run_until_complete(
                generate_plan(None, model, action_count))
            if is_workable_plan(client, plan):
                break
            print('Generated unworkable plan.  Trying again.')
    loop.close()
    with open(plan_file, 'w') as f:
        yaml.safe_dump(plan, f)


def run_glitch(plan_file, juju_model):
    with open(plan_file) as f:
        plan = yaml.safe_load(f)
    task = model.Task(command='glitch', args={'plan': plan_file, 'path': None})
    rule = model.Rule(task)
    loop = juju_model.loop
    for action in plan['actions']:
        logging.info('Performing action {}'.format(action))
        loop.run_until_complete(perform_action(action, juju_model, rule))


def execute_plan(plan_file, juju_data):
    client = client_for_existing(None, juju_data)
    client.wait_for_started()

    loop = asyncio.get_event_loop()
    with connected_model(loop, client) as juju_model:
        run_glitch(plan_file, juju_model)
    loop.close()
    client.wait_for_started()


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    kwargs = dict((k, v) for k, v in vars(args).items()
                  if k not in ('func', 'cmd'))
    args.func(**kwargs)
