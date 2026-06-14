import argparse
import json
from containerized_eval import eval_string_script 

def cli():
    args = argparse.ArgumentParser()
    args.add_argument("--lang", required=True, type=str, help="The language of the file")
    return args

if __name__ == "__main__":
    args = cli().parse_args() 
    prog_str = '''import scala.math._
import scala.collection.mutable._

object Problem {
    def exchange(lst1: List[Long], lst2: List[Long]): String = {
        val cntOdd = lst1.count(_ % 2 == 1)
        val cntEven = lst2.count(_ % 2 == 0)
        if (cntOdd <= cntEven) "YES" else "NO"
    }
    def main(args: Array[String]) = {
    assert(exchange((List[Long](1l.toLong, 2l.toLong, 3l.toLong, 4l.toLong)), (List[Long](1l.toLong, 2l.toLong, 3l.toLong, 4l.toLong))).equals(("YES")));
    assert(exchange((List[Long](1l.toLong, 2l.toLong, 3l.toLong, 4l.toLong)), (List[Long](1l.toLong, 5l.toLong, 3l.toLong, 4l.toLong))).equals(("NO")));
    assert(exchange((List[Long](1l.toLong, 2l.toLong, 3l.toLong, 4l.toLong)), (List[Long](2l.toLong, 1l.toLong, 4l.toLong, 3l.toLong))).equals(("YES")));
    assert(exchange((List[Long](5l.toLong, 7l.toLong, 3l.toLong)), (List[Long](2l.toLong, 6l.toLong, 4l.toLong))).equals(("YES")));
    assert(exchange((List[Long](5l.toLong, 7l.toLong, 3l.toLong)), (List[Long](2l.toLong, 6l.toLong, 3l.toLong))).equals(("NO")));
    assert(exchange((List[Long](3l.toLong, 2l.toLong, 6l.toLong, 1l.toLong, 8l.toLong, 9l.toLong)), (List[Long](3l.toLong, 5l.toLong, 5l.toLong, 1l.toLong, 1l.toLong, 1l.toLong))).equals(("NO")));
    assert(exchange((List[Long](100l.toLong, 200l.toLong)), (List[Long](200l.toLong, 200l.toLong))).equals(("YES")));
    }

}
'''
    # while True:
    #     try:
    #         line = input()
    #         prog_str += line + "\n"
    #     except EOFError:
    #         break
    print(json.dumps(eval_string_script(args.lang, prog_str)), end="")
