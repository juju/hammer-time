from unittest import TestCase
from unittest.mock import (
    call,
    patch,
    )

from jujupy.fake import fake_juju_client

from hammer_time.hammer_time import (
    cli_add_remove_many_container,
    cli_add_remove_many_machine,
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
        with patch.object(client._backend, 'juju',
                          wraps=client._backend.juju) as juju_mock:
            cli_add_remove_many_container(client)
        self.assertEqual([
            backend_call(client, 'add-machine', ('-n', '1')),
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
            backend_call(client, 'remove-machine', ('0',)),
            ], juju_mock.mock_calls)
