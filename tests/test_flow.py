import csv, os
import pytest
from fake_llm import FakeLLM
from voiceops.agent import SupportAgent
from voiceops.config import ensure_dataset_on_path


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    ensure_dataset_on_path()
    import tools.tools as t
    monkeypatch.setattr(t, "_TICKETS", str(tmp_path / "tickets.csv"))
    llm = FakeLLM()
    a = SupportAgent(llm=llm)
    a.fake, a.tickets_file = llm, tmp_path / "tickets.csv"
    return a


def order_block(a):
    sysm = last_system(a)
    return sysm.split("## VERIFIED ORDER RECORD(S)")[1].split("## POLICY CONTEXT")[0]


def last_system(a):
    return [c for c in a.fake.calls if "SESSION STATE" in c[0]][-1][0]


def test_pc002_flow_verification_gates_order_data(agent):
    s = agent.new_session("t", "text")
    r1 = agent.respond(s, "Hello, mujhe order ke baare mein help chahiye.")
    assert "NO ORDER IN CONTEXT" in last_system(agent) and "STY-1" in r1.sources
    r2 = agent.respond(s, "Order number hai 8978 2837.")
    sysm = last_system(agent)
    assert "ORDER ID RECEIVED (89782837)" in sysm and "order 89782837 (ACTIVE)" not in sysm  # nothing disclosed yet
    assert [c["name"] for c in r2.tool_calls] == ["lookup_order"] and r2.tool_calls[0]["args"] == {"order_id": "89782837"}
    assert "IDV-2" in r2.sources
    r3 = agent.respond(s, "Last four digits 4533 hain.")
    sysm = last_system(agent)
    assert "VERIFIED for order 89782837" in sysm and '"items"' in sysm
    ob = order_block(agent)
    assert "89782837" in ob and "delivery_pincode" not in ob and "customer_id" not in ob
    assert r3.tool_calls[0]["args"] == {"order_id": "89782837", "verification_value": "4533"}
    r4 = agent.respond(s, "Replacement approve kar do na? mujhe insaan se baat karni hai")
    assert r4.ticket_id and r4.ticket_id.startswith("TKT") and "ESC-1" in r4.sources
    rows = list(csv.DictReader(open(agent.tickets_file, encoding="utf-8")))
    assert rows[0]["order_id"] == "89782837" and "4533" not in rows[0]["summary"]  # secret scrubbed


def test_pc021_failed_verification_locks_after_one_retry(agent):
    s = agent.new_session("t2", "text")
    agent.respond(s, "ऑर्डर नंबर है 7746 1439।")
    r = agent.respond(s, "आख़िरी चार अंक 1266 हैं।")
    assert "attempt 1 of 2" in last_system(agent) and r.tool_calls[0]["args"]["verification_value"] == "1266"
    r = agent.respond(s, "आख़िरी चार अंक 1266 हैं।")
    assert "LOCKED" in last_system(agent) and r.tool_calls[0]["args"]["verification_value"] == "1266"
    n = len(s.tool_log)
    agent.respond(s, "आख़िरी चार अंक 1111 हैं।")        # third attempt: no tool call
    assert len(s.tool_log) == n and "LOCKED" in last_system(agent)
    assert not s.verified and order_block(agent).strip() == "none"


def test_pincode_verification_and_spoken_digits(agent):
    s = agent.new_session("t3", "voice")
    agent.respond(s, "the order number is one five nine one one one zero two")
    agent.respond(s, "pincode is four four zero zero one zero")
    assert "VERIFIED for order 15911102" in last_system(agent)


def test_guard_blocks_fake_cancellation(agent):
    agent.fake.replies = ["I have cancelled your order.", "You can cancel it free from My Orders in the app."]
    s = agent.new_session("t4", "text")
    r = agent.respond(s, "Please cancel it")
    assert r.guard and "cancelled" not in r.reply.lower() and "My Orders" in r.reply


def test_guard_blocks_fake_ticket_claim(agent):
    agent.fake.replies = ["Done, ticket has been created for you.", "Ticket has been created again."]
    s = agent.new_session("t5", "text")
    r = agent.respond(s, "I want compensation")
    assert r.guard and "colleague" in r.reply.lower()   # safe fallback, no false claim
