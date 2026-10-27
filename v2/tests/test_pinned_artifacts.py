import hashlib
import io

import pytest

from talkinghead_v2.assets.pinned import PinnedFile, provision, verify
from talkinghead_v2.contracts import ContractError


def artifact(data=b'approved-source'):
    return PinnedFile('geometry.ma',len(data),hashlib.sha256(data).hexdigest(),'https://example.org/geometry.ma')


def test_exact_download_is_immutable_and_offline_verify_has_no_fetch(tmp_path):
    entry=artifact()
    calls=[]
    def fetch(url):calls.append(url);return io.BytesIO(b'approved-source')
    destination=provision(tmp_path,'approved-revision',(entry,),fetch=fetch,reserve_bytes=0)
    assert len(calls)==1 and (destination/'geometry.ma').stat().st_mode & 0o777 == 0o444
    assert verify(destination,(entry,))==destination
    assert provision(tmp_path,'approved-revision',(entry,),fetch=fetch,reserve_bytes=0)==destination
    assert len(calls)==1


@pytest.mark.parametrize('content',[b'short',b'wrong-source!!!',b'approved-source-extra'])
def test_partial_wrong_or_oversize_never_publishes(tmp_path,content):
    with pytest.raises(ContractError):
        provision(tmp_path,'revision',(artifact(),),fetch=lambda _:io.BytesIO(content),reserve_bytes=0)
    assert not (tmp_path/'revision').exists()
    assert not list(tmp_path.glob('blobs/.download-*'))


def test_interrupted_snapshot_reuses_only_verified_blob(tmp_path):
    first,second=artifact(),PinnedFile('rig.json',2,hashlib.sha256(b'{}').hexdigest(),'https://example.org/rig.json')
    def failed(url):
        if url.endswith('rig.json'):raise OSError('interrupted download')
        return io.BytesIO(b'approved-source')
    with pytest.raises(OSError):provision(tmp_path,'revision',(first,second),fetch=failed,reserve_bytes=0)
    assert not (tmp_path/'revision').exists()
    called=[]
    def retry(url):called.append(url);return io.BytesIO(b'{}')
    destination=provision(tmp_path,'revision',(first,second),fetch=retry,reserve_bytes=0)
    assert called==[second.url] and verify(destination,(first,second))


@pytest.mark.parametrize('name',['../escape','/absolute','.','a/../b','a\\b','a//b'])
def test_path_escape_is_rejected(name):
    with pytest.raises(ContractError):PinnedFile(name,1,'a'*64,'https://example.org/item')


def test_snapshot_symlink_and_unexpected_payload_are_not_accepted(tmp_path):
    entry=artifact(); destination=provision(tmp_path,'revision',(entry,),fetch=lambda _:io.BytesIO(b'approved-source'),reserve_bytes=0)
    (destination/'unexpected.py').write_text('no execution')
    with pytest.raises(ContractError):verify(destination,(entry,))
    (destination/'unexpected.py').unlink()
    alias=tmp_path/'alias';alias.symlink_to(destination,target_is_directory=True)
    with pytest.raises(ContractError):verify(alias,(entry,))


def test_offline_and_disk_reserve_prevent_network(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    with pytest.raises(ContractError,match='offline'):provision(tmp_path,'revision',(artifact(),))
    monkeypatch.delenv('HF_HUB_OFFLINE')
    with pytest.raises(ContractError,match='reserve'):
        provision(tmp_path,'revision',(artifact(),),fetch=lambda _:pytest.fail('network called'),reserve_bytes=10**30)
