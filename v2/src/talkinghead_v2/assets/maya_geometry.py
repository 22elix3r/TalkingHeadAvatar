"""Data-only Maya ASCII mesh extraction. No MEL, plugins or references execute.

This deliberately narrow reader accepts contiguous point/edge arrays and simple
triangles/quads from an explicitly selected mesh node. Triangulation is an
explicit derived topology, preserving source vertex and polygon identities.
"""
from dataclasses import dataclass
import math
import re

from talkinghead_v2.contracts import ContractError

MAX_VERTICES = 100000
MAX_EDGES = 300000
MAX_POLYGONS = 150000
NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')


@dataclass(frozen=True)
class MayaGeometry:
    vertices: tuple
    triangles: tuple
    polygon_vertices: tuple
    triangle_source_polygons: tuple
    source_units: str


def node_body(text, name):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,100}', name): raise ContractError('explicit bounded mesh name required')
    nodes = list(re.finditer(r'^createNode\s+[^;]+;', text, re.MULTILINE))
    matches = [index for index, node in enumerate(nodes)
        if re.match(r'createNode mesh -n "' + re.escape(name) + r'"(?:\s|;)', node.group())]
    if len(matches) != 1: raise ContractError('selected mesh node must occur exactly once')
    index = matches[0]
    return text[nodes[index].end():nodes[index+1].start() if index+1<len(nodes) else len(text)]


def dense_array(body, attribute, width, bound, *, integers=False):
    declaration = re.findall(r'setAttr\s+-s\s+(\d+)\s+"\.' + attribute + r'"\s*;', body)
    if len(declaration) != 1 or not 1 <= int(declaration[0]) <= bound: raise ContractError('bounded declared array required: ' + attribute)
    count = int(declaration[0]); values = []
    pattern = r'setAttr\s+"\.' + attribute + r'\[(\d+)(?::(\d+))?\]"\s+([^;]+);'
    for match in re.finditer(pattern, body):
        first, last = int(match[1]), int(match[2] or match[1])
        if first != len(values) or last < first or last >= count: raise ContractError('array gap/overlap/extent mismatch: ' + attribute)
        tokens = match[3].split()
        if len(tokens) != (last-first+1)*width: raise ContractError('array tuple width mismatch: ' + attribute)
        if integers:
            if any(not re.fullmatch(r'-?\d+', token) for token in tokens): raise ContractError('integer array required')
            converted = [int(token) for token in tokens]
        else:
            if any(not NUMBER.fullmatch(token) for token in tokens): raise ContractError('numeric data required; scene commands are never evaluated')
            converted = [float(token) for token in tokens]
            if any(not math.isfinite(value) or abs(value)>1000000 for value in converted): raise ContractError('bounded finite points required')
        values.extend(tuple(converted[index:index+width]) for index in range(0,len(converted),width))
    if len(values) != count: raise ContractError('incomplete declared array: ' + attribute)
    return tuple(values)


def extract_geometry(text, name):
    if len(text)>100*1024**2 or '\0' in text: raise ContractError('bounded Maya ASCII source required')
    units = re.findall(r'^currentUnit -l (\w+) -a degree -t \w+;', text, re.MULTILINE)
    if units != ['centimeter']: raise ContractError('explicit Maya centimeter convention required')
    body = node_body(text, name)
    vertices = dense_array(body,'vt',3,MAX_VERTICES)
    edges = dense_array(body,'ed',3,MAX_EDGES,integers=True)
    for a,b,smooth in edges:
        if a==b or not 0<=a<len(vertices) or not 0<=b<len(vertices) or smooth not in (0,1): raise ContractError('invalid source edge; no clamping')
    declared = re.findall(r'setAttr\s+-s\s+(\d+)\s+-ch\s+\d+\s+"\.fc"\s*;',body)
    if len(declared)!=1 or not 1<=int(declared[0])<=MAX_POLYGONS: raise ContractError('bounded source polygon count required')
    polygons=[]
    pattern=r'setAttr\s+"\.fc\[(\d+)(?::(\d+))?\]"\s+(?:-type\s+"polyFaces"\s+)?([^;]+);'
    for match in re.finditer(pattern,body):
        first,last=int(match[1]),int(match[2] or match[1])
        if first!=len(polygons) or last<first or last>=int(declared[0]): raise ContractError('polygon gap/overlap/extent mismatch')
        if first==0 and '-type "polyFaces"' not in match[0]: raise ContractError('explicit first polygon-array type required')
        faces=[]
        for line in match[3].strip().splitlines():
            tokens=line.split()
            if not tokens:continue
            if tokens[0] in {'mu','mc','mf','mh'}:
                # UV/color indices are not geometry. Reject malformed lengths,
                # retain source geometry; this extractor makes no texture claim.
                if len(tokens)<3 or not all(re.fullmatch(r'-?\d+', value) for value in tokens[1:]) or len(tokens)!=3+int(tokens[2]):
                    raise ContractError('malformed source UV/color face annotation')
                continue
            if tokens[0]!='f' or len(tokens)<2 or tokens[1] not in {'3','4'} or len(tokens)!=2+int(tokens[1]):
                raise ContractError('only simple triangles/quads supported; holes and scene commands are rejected')
            if any(not re.fullmatch(r'-?\d+', value) for value in tokens[2:]): raise ContractError('invalid polygon edge token')
            directed=[]
            for value in map(int,tokens[2:]):
                index=value if value>=0 else -value-1
                if not 0<=index<len(edges): raise ContractError('polygon edge index out of range')
                a,b,_=edges[index];directed.append((a,b) if value>=0 else (b,a))
            if any(edge[1]!=directed[(index+1)%len(directed)][0] for index,edge in enumerate(directed)):
                raise ContractError('polygon is not a closed oriented edge loop')
            polygon=tuple(edge[0] for edge in directed)
            if len(set(polygon))!=len(polygon): raise ContractError('duplicate source polygon vertex')
            faces.append(polygon)
        if len(faces)!=last-first+1: raise ContractError('polygon range count mismatch')
        polygons.extend(faces)
    if len(polygons)!=int(declared[0]): raise ContractError('incomplete source polygon array')
    triangles=[];identities=[]
    for index,polygon in enumerate(polygons):
        # Explicit fixed diagonal of the source quad, never vertex truncation.
        for corner in range(1,len(polygon)-1):
            triangles.append((polygon[0],polygon[corner],polygon[corner+1]));identities.append(index)
    return MayaGeometry(vertices,tuple(triangles),tuple(polygons),tuple(identities),'centimeter')
