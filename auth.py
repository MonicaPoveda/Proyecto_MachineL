"""Authentication blueprint.

Provides login, registration, password recovery and the ``login_required``
decorator used to protect the Platform page. It also exposes a small CSRF
helper so every state-changing form is protected without extra dependencies.
"""

import hmac
import re
import secrets
from functools import wraps

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from db import create_user, get_user_by_email, get_user_by_id, set_user_password

auth_bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
RESET_TOKEN_MAX_AGE = 3600  # seconds


def _safe_next(target):
    """Allow only same-site relative redirect targets to prevent open redirects."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("home")


def get_csrf_token():
    """Return (and lazily create) the per-session CSRF token."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def validate_csrf():
    """Return whether the submitted CSRF token matches the session token."""
    submitted = request.form.get("_csrf_token", "")
    expected = session.get("_csrf_token", "")
    return bool(submitted) and bool(expected) and hmac.compare_digest(expected, submitted)


def login_required(view):
    """Redirect unauthenticated users to the login page, preserving the target."""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("auth.login", next=request.path))
        g.user = get_user_by_id(session["user_id"])
        return view(*args, **kwargs)

    return wrapped_view


def _reset_serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="password-reset")


@auth_bp.app_context_processor
def inject_csrf_token():
    return {"csrf_token": get_csrf_token}


@auth_bp.route("/login/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if not validate_csrf():
            abort(400)

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = get_user_by_email(email) if email else None

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.", "danger")
        else:
            session.clear()
            session["user_id"] = user["id"]
            session["_csrf_token"] = secrets.token_hex(32)
            return redirect(_safe_next(request.args.get("next")))

    return render_template("auth/login.html", next=request.args.get("next", ""))


@auth_bp.route("/register/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if not validate_csrf():
            abort(400)

        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        errors = []
        if not email:
            errors.append("Email is required.")
        elif len(email) > 254 or not EMAIL_RE.match(email):
            errors.append("Enter a valid email.")

        if not password:
            errors.append("Password is required.")
        elif len(password) < MIN_PASSWORD_LENGTH:
            errors.append(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
            )
        elif len(password) > MAX_PASSWORD_LENGTH:
            errors.append(
                f"Password must not exceed {MAX_PASSWORD_LENGTH} characters."
            )

        if not confirm:
            errors.append("You must confirm the password.")
        elif password and password != confirm:
            errors.append("Passwords do not match.")

        if not errors and get_user_by_email(email) is not None:
            errors.append("An account with that email already exists.")

        if errors:
            for error in errors:
                flash(error, "danger")
        else:
            create_user(email, generate_password_hash(password))
            flash("Account created successfully. You can now sign in.", "success")
            return redirect(url_for("auth.login"))

    return render_template("auth/register.html")


@auth_bp.route("/forgot-password/", methods=["GET", "POST"])
def forgot_password():
    dev_reset_url = None

    if request.method == "POST":
        if not validate_csrf():
            abort(400)

        email = request.form.get("email", "").strip()
        user = get_user_by_email(email) if email else None

        if user is not None:
            token = _reset_serializer().dumps(user["email"])
            dev_reset_url = url_for("auth.reset_password", token=token, _external=True)

        # Same message whether or not the account exists (avoids user enumeration).
        flash(
            "If the email is registered, you will receive a link to reset your password.",
            "info",
        )

    return render_template("auth/forgot_password.html", dev_reset_url=dev_reset_url)


@auth_bp.route("/reset-password/<token>/", methods=["GET", "POST"])
def reset_password(token):
    try:
        email = _reset_serializer().loads(token, max_age=RESET_TOKEN_MAX_AGE)
    except SignatureExpired:
        flash("The recovery link has expired. Request a new one.", "danger")
        return redirect(url_for("auth.forgot_password"))
    except BadSignature:
        flash("The recovery link is invalid.", "danger")
        return redirect(url_for("auth.forgot_password"))

    if request.method == "POST":
        if not validate_csrf():
            abort(400)

        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if len(password) < MIN_PASSWORD_LENGTH:
            flash(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
                "danger",
            )
        elif len(password) > MAX_PASSWORD_LENGTH:
            flash(
                f"Password must not exceed {MAX_PASSWORD_LENGTH} characters.",
                "danger",
            )
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        else:
            set_user_password(email, generate_password_hash(password))
            flash("Password updated. You can now sign in.", "success")
            return redirect(url_for("auth.login"))

    return render_template("auth/reset_password.html", token=token)


@auth_bp.route("/logout/")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("home"))
