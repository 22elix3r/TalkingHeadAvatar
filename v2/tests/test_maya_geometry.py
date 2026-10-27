"""Data extraction and source identity, without Maya/remote execution."""
import pytest
from talkinghead_v2.assets.maya_geometry import extract_geometry
from talkinghead_v2.contracts import ContractError

SOURCE='''//Maya ASCII 2026 scene
currentUnit -l centimeter -a degree -t film;
createNode mesh -n "head" -p "neutral";
setAttr -s 4 ".vt";
setAttr ".vt[0:1]" 0 0 0 1 0 0;
setAttr ".vt[2:3]" 1 1 0 0 1 0;
setAttr -s 4 ".ed";
setAttr ".ed[0:3]" 0 1 1 1 2 1 3 2 0 0 3 1;
setAttr -s 1 -ch 4 ".fc";
setAttr ".fc[0]" -type "polyFaces"
 f 4 0 1 -3 -4
 mu 0 4 0 1 2 3;
createNode transform -n "unrelated";
python("untrusted code must never execute");
'''


def test_signed_edges_preserve_winding_and_source_polygon_identity():
    result=extract_geometry(SOURCE,'head')
    assert result.vertices==((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.))
    assert result.polygon_vertices==((0,1,2,3),)
    assert result.triangles==((0,1,2),(0,2,3))
    assert result.triangle_source_polygons==(0,0) and result.source_units=='centimeter'


@pytest.mark.parametrize('old,new',[
 ('-l centimeter','-l meter'),
 ('".vt[2:3]"','".vt[1:2]"'),
 ('".vt[2:3]" 1 1 0 0 1 0','".vt[2:3]" 1 1 0'),
 ('".vt[0:1]" 0 0 0 1 0 0','".vt[0:1]" nan 0 0 1 0 0'),
 ('0 1 1 1 2 1','0 4 1 1 2 1'),
 ('f 4 0 1 -3 -4','f 4 0 1 2 -4'),
 ('f 4 0 1 -3 -4','f 4 0 1 -3 -99'),
 ('f 4 0 1 -3 -4','h 4 0 1 -3 -4'),
 ('mu 0 4 0 1 2 3','mu 0 3 0 1 2 3'),
 ('-type "polyFaces"','-type "python"'),
 ('setAttr -s 1 -ch 4 ".fc"','setAttr -s 2 -ch 8 ".fc"'),
])
def test_unsupported_or_malformed_source_fails_without_repair(old,new):
    with pytest.raises(ContractError):extract_geometry(SOURCE.replace(old,new),'head')


def test_duplicate_and_absent_mesh_names_are_never_guessed():
    with pytest.raises(ContractError):extract_geometry(SOURCE,'missing')
    with pytest.raises(ContractError):extract_geometry(SOURCE+SOURCE,'head')


def test_polygons_can_continue_the_explicit_first_array_type():
    source=SOURCE.replace('setAttr -s 1 -ch 4 ".fc"','setAttr -s 2 -ch 8 ".fc"')
    source=source.replace('createNode transform -n "unrelated";', 'setAttr ".fc[1]"\n f 4 -4 -3 1 0;\ncreateNode transform -n "unrelated";')
    # The deliberately malformed winding must still be rejected, even when type is inherited.
    with pytest.raises(ContractError):extract_geometry(source,'head')
    source=source.replace('f 4 -4 -3 1 0','f 4 0 1 -3 -4')
    result=extract_geometry(source,'head')
    assert result.triangle_source_polygons==(0,0,1,1)
