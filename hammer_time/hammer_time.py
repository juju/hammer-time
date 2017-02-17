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
import yaml

from matrix.bus import Bus
from matrix import model
from matrix.tasks.glitch.plan import generate_plan
from matrix.tasks.glitch.main import glitch


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


@contextmanager
def connected_model(loop, jujudata):
    controller_name = jujudata.current_controller()
    if controller_name == '':
        raise NoCurrentController(
            'Juju has no current controller at {}'.format(jujudata.path))
    controller = jujudata.controllers()[controller_name]
    try:
        endpoint = controller['api-endpoints'][0]
    except IndexError:
        raise NoEndpoints('Juju controller has no endppoints set.')
    cacert = controller.get('ca-cert')
    accounts = jujudata.accounts()[controller_name]
    username = accounts['user']
    password = accounts.get('password')
    macaroons = get_macaroons() if not password else None
    models = jujudata.models()[controller_name]
    model_uuid = models['models'][models['current-model']]['uuid']
    model = Model(loop)
    loop.run_until_complete(model.connect(
        endpoint, model_uuid, username, password, cacert, macaroons,
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


def make_plan(plan_file, juju_data, action_count):
    loop = asyncio.get_event_loop()
    juju_data_obj = SpecificJujuData(juju_data)
    with connected_model(loop, juju_data_obj) as model:
        plan = loop.run_until_complete(
            generate_plan(None, model, action_count))
    loop.close()
    with open(plan_file, 'w') as f:
        yaml.safe_dump(plan, f)


def run_glitch(plan_file, juju_model):
    """This function runs a specified glitch plan against a model.

    Exceptions are logged to glitch.log.

    It returns True if glitch ran successfully, False if exceptions were
    raised.
    """
    loop = juju_model.loop
    bus = Bus(loop=loop)
    suite = []

    class config:
        path = None

    task = model.Task(command='glitch', args={'plan': plan_file, 'path': None})

    rule = model.Rule(task)

    context = model.Context(loop, bus, suite, config, None)
    context.juju_model = juju_model
    try:
        plan = loop.run_until_complete(glitch(context, rule, task))
    except model.TestFailure as e:
        return False
    else:
        return True


def execute_plan(plan_file, juju_data):
    loop = asyncio.get_event_loop()
    with connected_model(loop, SpecificJujuData(juju_data)) as juju_model:
        run_glitch(plan_file, juju_model)
    loop.close()


def main():
    args = parse_args()
    kwargs = dict((k, v) for k, v in vars(args).items()
                  if k not in ('func', 'cmd'))
    args.func(**kwargs)
