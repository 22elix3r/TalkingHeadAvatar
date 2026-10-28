from pathlib import Path
import pytest


def stage(root,allowed,denied=()):
    from talkinghead_v2.host_image import RootfsStage
    return RootfsStage(root,allowed,denied)


def test_bundle_copies_only_selected_software_and_preserves_hash(tmp_path):
    software=tmp_path/'software';software.mkdir();source=software/'code.py';source.write_text('print(42)\n')
    output=tmp_path/'context';bundle=stage(output,[software]);bundle.copy_file(source)
    target=output/source.relative_to('/')
    assert target.read_bytes()==source.read_bytes()
    assert bundle.records[str(source)]['size']==source.stat().st_size
    assert len(bundle.records[str(source)]['sha256'])==64
    secret=tmp_path/'credentials';secret.write_text('secret')
    with pytest.raises(ValueError):bundle.copy_file(secret)


def test_cache_and_profile_content_are_excluded_even_under_a_selected_parent(tmp_path):
    source=tmp_path/'project';source.mkdir();private=source/'var';private.mkdir();secret=private/'runtime.json';secret.write_text('credential')
    bundle=stage(tmp_path/'context',[source],[private])
    with pytest.raises(ValueError):bundle.copy_file(secret)


def test_destination_symlink_cannot_escape_context(tmp_path):
    source=tmp_path/'software';source.mkdir();file=source/'module.py';file.write_text('code')
    output=tmp_path/'context';bundle=stage(output,[source])
    ancestor=output/source.relative_to('/');ancestor.parent.mkdir(parents=True);ancestor.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError):bundle.copy_file(file)
    assert file.read_text()=='code'


def test_symlink_cannot_import_an_unselected_source(tmp_path):
    source=tmp_path/'software';source.mkdir();secret=tmp_path/'secret';secret.write_text('credential')
    (source/'alias').symlink_to(secret)
    with pytest.raises(ValueError):stage(tmp_path/'context',[source]).copy_file(source/'alias')


def test_native_dependency_output_rejects_missing_libraries():
    from talkinghead_v2.host_image import parse_ldd
    assert parse_ldd('linux-vdso.so.1 (0x1)\nlibc.so.6 => /lib64/libc.so.6 (0x2)\n/lib64/ld-linux-x86-64.so.2 (0x3)')=={
        Path('/lib64/libc.so.6'),Path('/lib64/ld-linux-x86-64.so.2')}
    with pytest.raises(ValueError):parse_ldd('librequired.so => not found')


def test_transient_environment_lock_is_not_bundled_or_changed(tmp_path):
    software=tmp_path/'software';software.mkdir();lock=software/'.lock';lock.write_text('');lock.chmod(0o666)
    code=software/'code.py';code.write_text('pass\n');output=tmp_path/'context';bundle=stage(output,[software])
    bundle.copy_tree(software)
    assert str(lock) not in bundle.records and str(code) in bundle.records
    assert lock.stat().st_mode&0o777==0o666
