import subprocess

from cli_rb.workspace.container_worker import execute
from cli_rb.workspace.tools import FixtureBackend, make_read_tool, make_workspace_tools


def test_read_schema_rejects_500_before_backend_execution():
    class Backend:
        def read_file(self, *args):
            raise AssertionError('Invalid argument reached backend')
    tool = make_read_tool(Backend())
    schema = tool.args_schema.model_json_schema()['properties']
    assert schema['max_lines']['maximum'] == 400
    assert '400' in tool.invoke({'path':'a.py','max_lines':500})
    assert make_read_tool(FixtureBackend({'a.py':'hello'})).invoke({'path':'a.py'}) == '1: hello'


def test_navigation_order_paging_and_protection(tmp_path):
    for name in ['z.py','a.py','sub/b.py','hidden/private.py']:
        p=tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('needle\n',encoding='utf-8')
    def run(action,args):
        return execute(tmp_path,{'action':action,'args':args,'protected_paths':['hidden']})
    assert run('list',{}) == 'a.py\nz.py\nsub/b.py'
    assert run('find',{'pattern':'*b.py'}) == 'sub/b.py'
    assert run('list',{'limit':1,'offset':1}) == 'z.py\n[更多结果：offset=2]'
    assert run('search',{'query':'needle','path':'sub/b.py'}) == 'sub/b.py:1: needle'
    assert 'hidden' not in run('search',{'query':'needle'})


def test_search_timeout_does_not_claim_absence():
    class Backend:
        def search_files(self,*args):
            raise subprocess.TimeoutExpired('docker start',30)
    tool = next(t for t in make_workspace_tools(Backend()) if t.name=='search_files')
    assert '不能视为没有匹配' in tool.invoke({'query':'symbol'})


def test_larger_read_page_is_contiguous_and_bounded(tmp_path):
    import re
    (tmp_path/'short.py').write_text('\n'.join('x' for _ in range(450)))
    page=execute(tmp_path,{'action':'read','args':{'path':'short.py','max_lines':400}})
    assert '400: x' in page and '第 401 行' in page
    (tmp_path/'long.py').write_text('\n'.join('x'*100 for _ in range(400)))
    page=execute(tmp_path,{'action':'read','args':{'path':'long.py','max_lines':400}})
    next_line=int(re.search(r'第 (\d+) 行',page).group(1))
    assert 100 < next_line < 400 and len(page)<12100
    continuation=execute(tmp_path,{'action':'read','args':{'path':'long.py','start_line':next_line,'max_lines':400}})
    assert continuation.startswith(f'{next_line}: ')
