from app.utils.security import hash_password, verify_password, create_access_token, decode_token


def test_password_hash_roundtrip():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)
    assert not verify_password("wrong", h)


def test_jwt_roundtrip():
    t = create_access_token("alice", "admin")
    payload = decode_token(t)
    assert payload["sub"] == "alice"
    assert payload["role"] == "admin"


def test_jwt_invalid():
    assert decode_token("not.a.token") is None
