from typing import Dict, List
from unittest import mock
from unittest.mock import MagicMock

import pytest
from lightning_cloud.openapi import V1ListMembershipsResponse, V1ListSecretsResponse, V1Membership, V1Secret

from lightning_app.utilities.secrets import _names_to_ids


@pytest.mark.parametrize(
    "secret_names, expected",
    [
        ([], {}),
        (
            ["first-secret", "second-secret"],
            {"first-secret": "1234", "second-secret": "5678"},
        ),
    ],
)
@mock.patch("lightning_cloud.login.Auth.authenticate", MagicMock())
@mock.patch("lightning_app.utilities.network.LightningClient.secret_service_list_secrets")
@mock.patch("lightning_app.utilities.network.LightningClient.projects_service_list_memberships")
def test_names_to_ids(
    list_memberships: MagicMock,
    list_secrets: MagicMock,
    secret_names: List[str],
    expected: Dict[str, str],
):
    list_memberships.return_value = V1ListMembershipsResponse(memberships=[V1Membership(project_id="default-project")])
    list_secrets.return_value = V1ListSecretsResponse(
        secrets=[V1Secret(name="first-secret", id="1234"), V1Secret(name="second-secret", id="5678")]
    )

    assert _names_to_ids(secret_names) == expected
