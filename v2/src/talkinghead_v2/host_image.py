"""Explicit offline software staging for a host-qualified research API image."""
import hashlib
import os
from pathlib import Path
import re
import stat


class RootfsStage:
    def __init__(self,root,allowed,denied=()):
        root=Path(root)
        if os.path.lexists(root):raise ValueError('preserve previous image context')
        root.mkdir(parents=True,mode=0o755)
        self.root=root.resolve();self.allowed=tuple(Path(p).resolve() for p in allowed)
        self.denied=tuple(Path(p).resolve() for p in denied);self.records={}

    def _source(self,path):
        path=Path(os.path.normpath(str(Path(path).absolute())))
        resolved=path.resolve(strict=True)
        if not any(resolved==p or resolved.is_relative_to(p) for p in self.allowed) or any(
            resolved==p or resolved.is_relative_to(p) for p in self.denied):
            raise ValueError('source is outside the selected software inventory')
        return path,resolved

    def _target(self,path):
        target=self.root/path.relative_to('/')
        if not target.parent.resolve().is_relative_to(self.root):raise ValueError('image context destination escape')
        target.parent.mkdir(parents=True,exist_ok=True,mode=0o755)
        if not target.parent.resolve().is_relative_to(self.root):raise ValueError('image context destination escape')
        return target

    def copy_file(self,path):
        path,resolved=self._source(path);info=resolved.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid not in (0,os.getuid()) or info.st_mode&0o022:
            raise ValueError('owned, non-writable regular software file required')
        if str(path) in self.records:return self.records[str(path)]
        target=self._target(path);digest=hashlib.sha256();size=0
        with resolved.open('rb') as source,open(target,'xb',opener=lambda p,f:os.open(p,f|os.O_NOFOLLOW,0o644)) as output:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                output.write(chunk);digest.update(chunk);size+=len(chunk)
        after=resolved.stat()
        if (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or size!=info.st_size:
            raise ValueError('software changed during image staging')
        mode=0o755 if info.st_mode&0o111 else 0o644;target.chmod(mode)
        row={'kind':'file','size':size,'sha256':digest.hexdigest(),'mode':mode}
        self.records[str(path)]=row;return row

    def copy_tree(self,path):
        base,_=self._source(path)
        for directory,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git','include'))
            for name in list(dirs):
                source=Path(directory)/name
                if source.is_symlink():
                    self._source(source);link=os.readlink(source)
                    if Path(link).is_absolute() or '..' in Path(link).parts:raise ValueError('unsafe software directory alias')
                    target=self._target(source);target.symlink_to(link,target_is_directory=True)
                    self.records[str(source)]={'kind':'symlink','target':link};dirs.remove(name)
            for name in sorted(files):
                if name not in ('.lock','.gitignore') and not name.endswith(('.pyc','.pyo')):self.copy_file(Path(directory)/name)


def parse_ldd(value):
    if 'not found' in value:raise ValueError('native software dependency unavailable')
    result=set()
    for line in value.splitlines():
        match=re.search(r'(?:=>\s*)?(/[^\s]+)\s+\(',line)
        if match:result.add(Path(match.group(1)))
    return result
