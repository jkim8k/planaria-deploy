#!/usr/bin/env python3
"""SRT 열차 검색/예약/일괄예약/조회/취소 스킬.
Usage: python run.py '{"action": "search", "dep": "평택지제", "arr": "수서", "date": "20260405", "time": "070000"}'
       python run.py '{"action": "reserve", "dep": "평택지제", "arr": "수서", "date": "20260405", "time": "070000", "train_index": 0}'
       python run.py '{"action": "batch_reserve", "dep": "평택지제", "arr": "수서", "date": "20260408", "time": "070000", "target_time": "0758", "days": 30}'
       python run.py '{"action": "my_reservations"}'
       python run.py '{"action": "cancel", "reservation_number": 12345}'
"""
import json, sys
from pathlib import Path
from datetime import datetime, timedelta

def load_creds():
    ws = Path(__file__).parent.parent.parent
    cfg = json.loads(ws.joinpath("config.json").read_text(encoding="utf-8"))
    srt_id = cfg["connectors"]["srt_id"]
    srt_pw = cfg["connectors"]["srt_pw"]
    if str(srt_pw).startswith("@secret:"):
        secret_key = str(srt_pw).split("@secret:", 1)[1]
        secrets_path = ws / "secrets.json"
        if secrets_path.exists():
            secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
            srt_pw = secrets.get(secret_key, srt_pw)
    return srt_id, srt_pw

def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    try:
        args = json.loads(raw)
    except Exception:
        print(json.dumps({"ok": False, "error": "invalid JSON args"}))
        return

    action = args.get("action", "search")

    try:
        from SRT import SRT, SeatType
    except ImportError:
        print(json.dumps({"ok": False, "error": "SRTrain not installed. Run: uv pip install SRTrain"}))
        return

    srt_id, srt_pw = load_creds()

    try:
        srt = SRT(srt_id, srt_pw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"SRT login failed: {e}"}, ensure_ascii=False))
        return

    try:
        if action == "search":
            dep = args.get("dep", "수서")
            arr = args.get("arr", "부산")
            date = args.get("date")
            time_ = args.get("time", "000000")
            trains = srt.search_train(dep, arr, date=date, time=time_, available_only=args.get("available_only", True))
            result = []
            for i, t in enumerate(trains[:10]):
                result.append({
                    "index": i, "train_number": t.train_number,
                    "dep": t.dep_station_name, "arr": t.arr_station_name,
                    "dep_date": t.dep_date, "dep_time": t.dep_time, "arr_time": t.arr_time,
                    "general_seat": "예약가능" if t.general_seat_available() else "매진",
                    "special_seat": "예약가능" if t.special_seat_available() else "매진",
                })
            print(json.dumps({"ok": True, "action": "search", "count": len(result), "trains": result}, ensure_ascii=False))

        elif action == "reserve":
            dep = args.get("dep", "수서")
            arr = args.get("arr", "부산")
            date = args.get("date")
            time_ = args.get("time", "000000")
            idx = int(args.get("train_index", 0))
            seat = args.get("seat_type", "GENERAL_FIRST")
            seat_map = {"GENERAL_FIRST": SeatType.GENERAL_FIRST, "GENERAL_ONLY": SeatType.GENERAL_ONLY,
                        "SPECIAL_FIRST": SeatType.SPECIAL_FIRST, "SPECIAL_ONLY": SeatType.SPECIAL_ONLY}
            trains = srt.search_train(dep, arr, date=date, time=time_)
            if idx >= len(trains):
                print(json.dumps({"ok": False, "error": f"train_index {idx} out of range ({len(trains)} trains)"}, ensure_ascii=False))
                return
            reservation = srt.reserve(trains[idx], special_seat=seat_map.get(seat, SeatType.GENERAL_FIRST))
            print(json.dumps({"ok": True, "action": "reserve", "reservation_number": reservation.reservation_number,
                              "total_cost": reservation.total_cost, "seat_count": reservation.seat_count,
                              "train": f"{reservation.train_name} {reservation.dep_station_name}→{reservation.arr_station_name}",
                              "dep_time": reservation.dep_time}, ensure_ascii=False))

        elif action == "batch_reserve":
            dep = args.get("dep", "수서")
            arr = args.get("arr", "부산")
            start_date_str = args.get("date")
            time_ = args.get("time", "070000")
            target_time = args.get("target_time", "")  # e.g. "0758"
            days = int(args.get("days", 30))
            weekdays_only = bool(args.get("weekdays_only", False))
            weekends_only = bool(args.get("weekends_only", False))

            if start_date_str:
                start = datetime.strptime(start_date_str, "%Y%m%d")
            else:
                start = datetime.now() + timedelta(days=1)

            reserved = []
            sold_out = []
            errors = []

            for offset in range(days):
                d = start + timedelta(days=offset)
                if weekdays_only and d.weekday() >= 5:  # 5=Sat, 6=Sun
                    continue
                if weekends_only and d.weekday() < 5:  # 0=Mon..4=Fri
                    continue
                ds = d.strftime("%Y%m%d")
                dl = d.strftime("%Y-%m-%d")
                try:
                    trains = srt.search_train(dep, arr, date=ds, time=time_, available_only=False)
                except Exception as e:
                    err_str = str(e)
                    if "발매개시준비중" in err_str or "WRD000052" in err_str:
                        break  # no more dates available
                    errors.append({"date": dl, "error": err_str[:200]})
                    continue

                target = None
                for t in trains:
                    if target_time and t.dep_time.startswith(target_time):
                        target = t
                        break
                if not target and not target_time and trains:
                    target = trains[0]

                if not target:
                    sold_out.append({"date": dl, "reason": "열차 없음"})
                    continue

                if not target.general_seat_available() and not target.special_seat_available():
                    sold_out.append({"date": dl, "train": target.train_number, "dep_time": target.dep_time})
                    continue

                seat_type = SeatType.GENERAL_FIRST if target.general_seat_available() else SeatType.SPECIAL_FIRST
                try:
                    r = srt.reserve(target, special_seat=seat_type)
                    reserved.append({
                        "date": dl, "train_number": target.train_number,
                        "dep_time": target.dep_time, "arr_time": target.arr_time,
                        "reservation_number": r.reservation_number,
                        "cost": r.total_cost,
                        "seat": "일반석" if target.general_seat_available() else "특실",
                    })
                except Exception as e:
                    errors.append({"date": dl, "error": str(e)[:200]})

            print(json.dumps({
                "ok": True, "action": "batch_reserve",
                "summary": f"예약 {len(reserved)}건, 매진 {len(sold_out)}건, 오류 {len(errors)}건",
                "reserved": reserved, "sold_out": sold_out, "errors": errors,
            }, ensure_ascii=False))

        elif action == "my_reservations":
            reservations = srt.get_reservations()
            result = []
            for r in reservations:
                result.append({
                    "reservation_number": r.reservation_number, "total_cost": r.total_cost,
                    "seat_count": r.seat_count,
                    "train": f"{r.train_name} {r.dep_station_name}→{r.arr_station_name}",
                    "dep_date": r.dep_date, "dep_time": r.dep_time, "paid": r.paid,
                })
            print(json.dumps({"ok": True, "action": "my_reservations", "count": len(result), "reservations": result}, ensure_ascii=False))

        elif action == "cancel":
            rnum = args.get("reservation_number")
            if not rnum:
                print(json.dumps({"ok": False, "error": "reservation_number required"}))
                return
            srt.cancel(int(rnum))
            print(json.dumps({"ok": True, "action": "cancel", "cancelled": rnum}, ensure_ascii=False))

        else:
            print(json.dumps({"ok": False, "error": f"unknown action: {action}. Use: search, reserve, batch_reserve, my_reservations, cancel"}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:500]}, ensure_ascii=False))
    finally:
        try:
            srt.logout()
        except Exception:
            pass

if __name__ == "__main__":
    main()
