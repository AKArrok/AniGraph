"""Alias 三级级联缓存的行为测试（不打真 LLM/网络）"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class AliasLayeredTest(unittest.TestCase):
    def setUp(self):
        # 每个 case 用独立临时缓存文件，避免污染真实 data/alias_cache.json
        self.tmpdir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmpdir, "alias_cache.json")

        import config
        self._orig_path = getattr(config, "ALIAS_CACHE_PATH", "")
        config.ALIAS_CACHE_PATH = self.cache_path

        # 重置模块级状态
        import agents.alias as alias_mod
        alias_mod._LEARNED_CACHE.clear()
        alias_mod._CACHE_LOADED = False
        self.alias_mod = alias_mod

    def tearDown(self):
        import config
        config.ALIAS_CACHE_PATH = self._orig_path
        # 清空 metadata_cache 避免跨用例污染
        try:
            from agents.cache import metadata_cache
            metadata_cache._alias_cache.clear()
        except Exception:
            pass

    def test_seed_hit_zero_llm(self):
        """L1 seed 命中不应调用 LLM 或 Web"""
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias") as llm_mock, \
             mock.patch.object(self.alias_mod, "_web_resolve_alias") as web_mock:
            result = self.alias_mod.resolve_alias_ex("素晴", use_web=True)
        self.assertEqual(result["full_name"], "为美好的世界献上祝福！")
        self.assertEqual(result["source"], "cache")
        llm_mock.assert_not_called()
        web_mock.assert_not_called()

    def test_l2_llm_hit_writes_cache_and_disk(self):
        """L2 LLM 命中后写回 learned cache + 磁盘"""
        from agents.alias import AliasLLMOutput
        mocked = AliasLLMOutput(anime="虚构番剧A", confidence=0.85)
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias", return_value=mocked) as llm_mock, \
             mock.patch.object(self.alias_mod, "_web_resolve_alias") as web_mock:
            result = self.alias_mod.resolve_alias_ex("虚构缩写A", use_web=True)

        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["full_name"], "虚构番剧A")
        llm_mock.assert_called_once()
        web_mock.assert_not_called()

        # 内存 cache 命中
        self.assertIn("虚构缩写a", self.alias_mod._LEARNED_CACHE)
        # 磁盘 cache 落盘
        self.assertTrue(os.path.exists(self.cache_path))
        with open(self.cache_path, "r", encoding="utf-8") as f:
            disk = json.load(f)
        self.assertIn("虚构缩写a", disk)
        self.assertEqual(disk["虚构缩写a"]["full_name"], "虚构番剧A")

    def test_l2_second_call_hits_cache(self):
        """L2 写入后第二次调用应从 cache 命中，零 LLM 调用"""
        from agents.alias import AliasLLMOutput
        mocked = AliasLLMOutput(anime="虚构番剧B", confidence=0.9)
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias", return_value=mocked) as llm_mock:
            first = self.alias_mod.resolve_alias_ex("虚构缩写B", use_web=False)
            second = self.alias_mod.resolve_alias_ex("虚构缩写B", use_web=False)
        self.assertEqual(first["source"], "llm")
        self.assertEqual(second["source"], "cache")
        llm_mock.assert_called_once()

    def test_l3_web_only_when_l2_low_confidence(self):
        """L2 低置信度时才走 L3；L2 高置信度绝不打 Web"""
        from agents.alias import AliasLLMOutput
        low = AliasLLMOutput(anime="", confidence=0.0)
        web_hit = AliasLLMOutput(anime="虚构番剧C", confidence=0.8)
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias", return_value=low), \
             mock.patch.object(self.alias_mod, "_web_resolve_alias", return_value=web_hit) as web_mock, \
             mock.patch("config.ENABLE_WEB_SEARCH", True), \
             mock.patch("config.TAVILY_API_KEY", "fake"):
            result = self.alias_mod.resolve_alias_ex("虚构冷门C", use_web=True)

        self.assertEqual(result["source"], "web")
        self.assertEqual(result["full_name"], "虚构番剧C")
        web_mock.assert_called_once()

    def test_l3_disabled_when_use_web_false(self):
        """use_web=False 时 L2 失败后不应触发 Web"""
        from agents.alias import AliasLLMOutput
        low = AliasLLMOutput(anime="", confidence=0.0)
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias", return_value=low), \
             mock.patch.object(self.alias_mod, "_web_resolve_alias") as web_mock:
            result = self.alias_mod.resolve_alias_ex("完全未知词", use_web=False)
        self.assertEqual(result["source"], "miss")
        web_mock.assert_not_called()

    def test_low_confidence_not_persisted(self):
        """confidence < _MIN_CACHE_CONFIDENCE 时不写磁盘（避免污染）"""
        from agents.alias import AliasLLMOutput
        mid = AliasLLMOutput(anime="低信心番剧", confidence=0.55)
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias", return_value=mid):
            result = self.alias_mod.resolve_alias_ex("模糊词", use_web=False)
        self.assertEqual(result["source"], "llm")
        # 内存有，磁盘无
        self.assertIn("模糊词", self.alias_mod._LEARNED_CACHE)
        self.assertFalse(os.path.exists(self.cache_path),
                          "低置信度不应写磁盘")

    def test_disk_cache_loaded_on_startup(self):
        """启动时从磁盘加载既有 alias_cache.json"""
        # 预置一份磁盘缓存
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "预置词": {"full_name": "预置番剧X", "confidence": 0.9, "source": "llm", "ts": 0}
            }, f)

        # 强制重新加载
        self.alias_mod._LEARNED_CACHE.clear()
        self.alias_mod._CACHE_LOADED = False

        with mock.patch.object(self.alias_mod, "_llm_resolve_alias") as llm_mock:
            result = self.alias_mod.resolve_alias_ex("预置词", use_web=False)
        self.assertEqual(result["source"], "cache")
        self.assertEqual(result["full_name"], "预置番剧X")
        llm_mock.assert_not_called()

    def test_resolve_alias_dict_still_works(self):
        """向后兼容：resolve_alias_dict 依然只走 L1"""
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias") as llm_mock:
            # re0 -> SQLite Alias 表命中 "Re：从零开始的异世界生活"（全角冒号，Bangumi 规范形）
            result = self.alias_mod.resolve_alias_dict("re0")
            self.assertIsNotNone(result)
            self.assertIn("从零开始", result)
            # 命中 seed 保留的圈内简称
            self.assertEqual(self.alias_mod.resolve_alias_dict("巨人"), "进击的巨人")
            self.assertIsNone(self.alias_mod.resolve_alias_dict("完全无关词zzzzz"))
        llm_mock.assert_not_called()

    # ── 属性词剥离 ──────────────────────────────────────────

    def test_attr_suffix_strip_hits_seed(self):
        """L1 剥离属性词后应能命中 seed（不打 LLM）"""
        cases = [
            ("素晴怎么样", "为美好的世界献上祝福！"),
            ("巨人评分多少", "进击的巨人"),
            ("石头门声优", "命运石之门"),      # SQLite Alias 命中
            ("钢炼几集", "钢之炼金术师 FULLMETAL ALCHEMIST"),
            ("赛马娘好看吗", "赛马娘 Pretty Derby"),  # SQLite Alias
            ("间谍过家家评分", "间谍过家家"),
            ("素晴呢", "为美好的世界献上祝福！"),
        ]
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias") as llm_mock, \
             mock.patch.object(self.alias_mod, "_web_resolve_alias") as web_mock:
            for q, expected in cases:
                res = self.alias_mod.resolve_alias_ex(q, use_web=False)
                self.assertEqual(res["source"], "cache",
                                  msg=f"query={q!r} 应命中 L1，实际 {res}")
                self.assertEqual(res["full_name"], expected, msg=f"query={q!r}")
        llm_mock.assert_not_called()
        web_mock.assert_not_called()

    def test_attr_strip_llm_uses_stripped_query(self):
        """L2 miss 后再学时，喂给 LLM 的应该是剥离后的短查询，cache key 也归一化"""
        from agents.alias import AliasLLMOutput
        seen_calls = []

        def fake_llm(q):
            seen_calls.append(q)
            return AliasLLMOutput(anime="虚构番剧Z", confidence=0.9)

        with mock.patch.object(self.alias_mod, "_llm_resolve_alias", side_effect=fake_llm):
            res1 = self.alias_mod.resolve_alias_ex("未知词ZZZ评分", use_web=False)
            res2 = self.alias_mod.resolve_alias_ex("未知词ZZZ多少集", use_web=False)

        # LLM 只应被调用一次；第二次因为 cache key 归一到剥离后，命中 cache
        self.assertEqual(len(seen_calls), 1, msg=f"实际调用: {seen_calls}")
        self.assertEqual(seen_calls[0], "未知词zzz",
                          msg=f"应用剥离后的短查询，实际: {seen_calls[0]!r}")
        self.assertEqual(res1["source"], "llm")
        self.assertEqual(res2["source"], "cache")

    def test_attr_strip_does_not_degrade(self):
        """剥离后过短/纯数字/退化的不应生效，避免误命中"""
        # 纯属性词/纯数字，剥离后为空或过短，L1 应 miss（不应把"评分"错剥成""再瞎匹配）
        for q in ["评分", "怎么样", "几集"]:
            with mock.patch.object(self.alias_mod, "_llm_resolve_alias",
                                     return_value=None):
                res = self.alias_mod.resolve_alias_ex(q, use_web=False)
            self.assertEqual(res["source"], "miss",
                              msg=f"query={q!r} 不该命中，实际 {res}")

    def test_query_prefix_strip_hits_seed(self):
        """L1 剥离定义/介绍类前缀后应命中 seed（不打 LLM）"""
        cases = [
            ("什么是钢炼", "钢之炼金术师 FULLMETAL ALCHEMIST"),
            ("介绍一下钢炼", "钢之炼金术师 FULLMETAL ALCHEMIST"),
            ("介绍一下钢炼几集", "钢之炼金术师 FULLMETAL ALCHEMIST"),  # 前缀+后缀同剥
        ]
        with mock.patch.object(self.alias_mod, "_llm_resolve_alias") as llm_mock, \
             mock.patch.object(self.alias_mod, "_web_resolve_alias") as web_mock:
            for q, expected in cases:
                res = self.alias_mod.resolve_alias_ex(q, use_web=False)
                self.assertEqual(res["source"], "cache",
                                  msg=f"query={q!r} 应命中 L1，实际 {res}")
                self.assertEqual(res["full_name"], expected, msg=f"query={q!r}")
        llm_mock.assert_not_called()
        web_mock.assert_not_called()

    def test_intent_guard_blocks_entity_lock(self):
        """意图守卫：推荐/类似类查询不做确定性实体锁定，交给下游语义检索。

        即便句子里含有明确番名（钢炼/进击的巨人），也必须返回 miss，
        避免把"有没有类似X的番"劫持成"查 X"这部番的事实问答。
        """
        intent_queries = [
            "有没有类似钢炼的番",
            "有没有类似进击的巨人的番",
            "推荐几部像钢炼一样的番",
            "求推钢炼这种的",
            "找番 进击的巨人之类的",
        ]
        for q in intent_queries:
            res = self.alias_mod.resolve_alias_dict(q)
            self.assertIsNone(
                res, msg=f"query={q!r} 属推荐意图，不应锁定实体，实际命中 {res!r}")


    def test_substring_no_false_positive(self):
        """对抗集：短别名/低覆盖子串不应劫持无关查询（回归 seed 包含匹配裸奔的 bug）"""
        adversarial = [
            "滚开", "大蒜怎么做", "这个op真好听", "click here",
            "刀剑乱舞", "无职者怎么办", "86版电影", "2077游戏好玩吗",
            "果家菜价格", "物语作文",
        ]
        for q in adversarial:
            res = self.alias_mod.resolve_alias_dict(q)
            self.assertIsNone(
                res, msg=f"query={q!r} 不应被 seed 子串劫持，实际命中 {res!r}")

    def test_substring_positive_recall(self):
        """正样本：正常带后缀/长别名查询仍应命中，确认安全闸没误伤召回"""
        positive = {
            "素晴怎么样": "为美好",
            "钢炼几集": "钢之炼金",
            "孤独摇滚好看吗": "孤独摇滚",
            "紫罗兰永恒花园": "紫罗兰",
            "咒回": "咒术",
        }
        for q, expect in positive.items():
            res = self.alias_mod.resolve_alias_dict(q)
            self.assertIsNotNone(res, msg=f"query={q!r} 应命中却 miss")
            self.assertIn(expect, res,
                          msg=f"query={q!r} 期望含 {expect!r}，实际 {res!r}")


    def test_seed_targets_align_official_titles(self):
        """seed 目标名必须对齐官方 SQLite 标题（消除双规范名）。

        白名单是官方库确实没有的作品（seed 作唯一来源），其余 seed 目标
        都应精确等于某个 Anime.anime_title，否则下游查元数据会 miss。
        """
        import sqlite3, config
        db = getattr(config, "ALIAS_DB_PATH", "")
        if not db or not os.path.exists(db):
            self.skipTest("SQLite 别名库不存在，跳过对齐校验")
        conn = sqlite3.connect(db)
        titles = set(r[0] for r in conn.execute(
            "SELECT anime_title FROM Anime WHERE anime_title IS NOT NULL"))
        conn.close()
        # 官方库确无、seed 作唯一来源的合法例外
        whitelist = {"ONE PIECE", "物语系列", "进击的巨人 The Final Season"}
        offenders = []
        for alias, target in self.alias_mod.HARDCODED_ALIASES.items():
            if target in whitelist:
                continue
            if target not in titles:
                offenders.append((alias, target))
        self.assertEqual(
            offenders, [],
            msg=f"seed 目标未对齐官方标题（双规范名风险）: {offenders}")

    def test_haruhen_series_canonical(self):
        """春物/果青/大老师 应解成官方无句号规范名，分季各自对齐"""
        cases = {
            "春物": "我的青春恋爱物语果然有问题",
            "果青": "我的青春恋爱物语果然有问题",
            "大老师": "我的青春恋爱物语果然有问题",
        }
        for q, expect in cases.items():
            r = self.alias_mod.resolve_alias_ex(q, use_web=False)
            self.assertEqual(r["full_name"], expect,
                             msg=f"{q!r} 期望 {expect!r}，实际 {r['full_name']!r}")


if __name__ == "__main__":
    unittest.main()
