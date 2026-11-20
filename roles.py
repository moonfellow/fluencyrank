# role vocabulary for the contract rosters.
#
# database/processed/people.csv gives every named party a role string, comma
# separated when one person holds several. the vocabulary is open: ~1700
# distinct tokens over 22k rows, most of them one-offs, and it mixes
# orthographies (traditional/simplified, 場/塲/场, 憑/凴/凭, 戶/户) with
# scribal typos the transcribers kept in brackets (在傷[場]).
#
# so the tokens are matched by pattern, not enumerated. ROLE_FAMILIES groups
# them into the functional roles the analysis cares about; SUBTYPES splits the
# three families that are worth reading in detail. order matters in both:
# the first pattern that hits wins for subtypes, and own_hand is tested before
# scribe so 親筆 ("in his own hand") is not swept in with 代筆.
#
# coverage on the plausible-match roster is ~99% of role tokens. the residue
# is one-off kin terms and oddities (營業者, 天開公嗣孫).

import re

# (key, label, pattern). a token may belong to several families: "兄,買人"
# is kin and buyer, and 在場憑中 is one witness token, not two.
ROLE_FAMILIES = [
    ("own_hand", "親筆",
     r"親筆|亲笔|親書|亲书"),
    ("scribe", "代筆",
     r"代筆|代笔|代書|代书|代字|^的?筆$|^記$|^手$"),
    ("witness", "witness / middleman",
     r"在見|在见|在場|在塲|在场|在傷|在腸|憑中|凴中|凭中|見中|见中|原中|中人|"
     r"中見|中见|說合中|说合中|^見|^见|證人|証人|証明|证明|^平人$|托中|找中|"
     r"契中|憑人|凴人|見$|^中$|^仝?中$|保人|酌議"),
    ("initiator", "立契人",
     r"^立|^出與|^出与|^出票|^出典|^仝?賣人?$|^賣主$|^售賣人$|^轉賣人$"),
    ("buyer", "buyer / taker",
     r"買人|買主|买人|承買|承买|受業|受业|^受主$|^受人$|^受租人$|銀主|银主|"
     r"錢主|钱主|債主|典主|承找|承當|承当|承退|承批|承拚|承拼|承典|承換|承换|"
     r"承管|承抽|抵換|換人|換業"),
    ("owner", "landholder / household head",
     r"業主|業戶|業户|业主|田主|山主|糧戶|粮户|粮戶|花戶|花户|戶名|户名|戶主|"
     r"户主|管業|管业|粮主|佃戶|佃户"),
    ("tax", "tax / registry",
     r"稅|税|推戶|推户|收戶|收户|推收|經收|经收|經徵|經征|经征|繳捐|缴捐"),
    ("official", "official / headman",
     r"正堂|知事|縣長|县长|知縣|知县|保長|保长|地保|鄰長|邻长|族長|族长|經承|"
     r"经承|村長|村长"),
    ("kin", "family term",
     r"(胞|堂|族|房|嫡|本家|親|姻|賢|贤|長|长|次|三|四|大|二|滿|满)?"
     r"(兄弟|兄|弟|叔|伯|侄|姪|父|母|子|男|孫|孙|婿|舅|姊|妹|公|郎|娶)"),
]

FOCUS_FAMILIES = ["scribe", "witness", "initiator"]

# per-family subtypes, first match wins. the initiator patterns are ordered by
# how specific the transaction verb is: 立杜找断截田契人 is a 找 (supplementary
# payment on an earlier sale), not a 賣, so 找 is tested before 賣.
SUBTYPES = {
    "scribe": [
        ("dictated", "依口代筆", r"依口"),
        ("daishu", "代書/代字", r"代書|代书|代字"),
        ("daibi", "代筆", r"代筆|代笔"),
        ("other", "other writing hand", r"."),
    ],
    "witness": [
        ("zaijian", "在見", r"在見|在见"),
        ("zaichang", "在場", r"在場|在塲|在场|在傷|在腸"),
        ("pingzhong", "憑中", r"憑中|凴中|凭中|憑人|凴人"),
        ("jianzhong", "見中", r"見中|见中"),
        ("yuanzhong", "原中", r"原中"),
        ("shuohe", "說合中", r"說合|说合"),
        ("zhongren", "中人", r"中人|中見|中见|托中|找中|契中|^中$"),
        ("jian_act", "見找/見當/…", r"^見|^见"),
        ("zhengren", "證人", r"證|証|证"),
        ("other", "other witness", r"."),
    ],
    "initiator": [
        ("zhao", "找", r"找"),
        ("mai", "賣", r"賣|卖|絕賣|断|斷"),
        ("dang", "當·典", r"當|当|典"),
        ("fen", "分", r"分|拍"),
        ("pi", "批·拚·佃", r"批|拚|拼|佃|耕|租"),
        ("piao", "送·討·劄·票", r"送|討|讨|劄|剳|票|推"),
        ("jie", "借", r"借"),
        ("shou", "收", r"收|領|领"),
        ("tui", "退", r"退"),
        ("huan", "換", r"換|换"),
        ("yue", "約·議", r"約|约|議|议|合同|關|関"),
        ("chuyu", "出與", r"出與|出与|撥|拨|抽|接管|托管"),
        ("other", "other contracting party", r"."),
    ],
}

FAMILY_LABEL = {k: l for k, l, _ in ROLE_FAMILIES}
_FAMILY_RE = [(k, re.compile(p)) for k, _, p in ROLE_FAMILIES]
_SUBTYPE_RE = {f: [(k, l, re.compile(p)) for k, l, p in subs]
               for f, subs in SUBTYPES.items()}
SUBTYPE_LABEL = {f: {k: l for k, l, _ in subs} for f, subs in SUBTYPES.items()}

SPLIT_RE = re.compile(r"[,，/、]")


def split_roles(text):
    return [t.strip() for t in SPLIT_RE.split(text or "") if t.strip()]


def token_families(token):
    return [k for k, r in _FAMILY_RE if r.search(token)]


def token_subtype(family, token):
    for key, _, r in _SUBTYPE_RE.get(family, []):
        if r.search(token):
            return key
    return "other"
