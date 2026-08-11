#!/usr/bin/env python3
"""
Submit VALD3 "Extract Stellar" requests for the atomic line lists used to
synthesize intrinsic PySME stellar templates in atmo-retrieval.

Prerequisites
-------------
1. Register/login at VALD3 with your email address.
2. Install Selenium:
       python -m pip install selenium
3. Have Firefox installed.

This script sets the VALD Unit selection itself at startup:
       Energy unit: eV
       Medium: vacuum
       Wavelength unit: angstrom
       VdW syntax: default

The script intentionally does only the request submission. It does not download
or parse the returned VALD files.

Examples
--------
Test only KELT-20 first:
    python request_vald_pysme_linelist.py --email you@example.com --only kelt20b

Submit all line lists:
    python request_vald_pysme_linelist.py --email you@example.com

Skip the interactive confirmation:
    python request_vald_pysme_linelist.py --email you@example.com --yes
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


VALD_URL = "https://vald.astro.uu.se/vald.php"
WAIT_SECONDS = 20
BETWEEN_REQUESTS_SECONDS = 2.0


@dataclass(frozen=True)
class Target:
    start: float
    end: float
    threshold: float
    vmicro: float
    teff: int
    logg: float
    mh: float
    comment: str


TARGETS: dict[str, Target] = {
    "kelt20b": Target(
        start=4200,
        end=7500,
        threshold=0.02,
        vmicro=2.8,
        teff=8940,
        logg=4.32,
        mh=-0.02,
        comment="atmo-retrieval PySME line list; KELT-20/MASCARA-2; vacuum; 4200-7500 A",
    ),
    "kelt9b": Target(
        start=4700,
        end=7500,
        threshold=0.02,
        vmicro=2.3,
        teff=9560,
        logg=4.08,
        mh=+0.16,
        comment="atmo-retrieval PySME line list; KELT-9; vacuum; 4700-7500 A",
    ),
    "toi1518b": Target(
        start=4700,
        end=7500,
        threshold=0.02,
        vmicro=3.0,
        teff=7810,
        logg=4.19,
        mh=+0.23,
        comment="atmo-retrieval PySME line list; TOI-1518; vacuum; 4700-7500 A",
    ),
    "toi1431b": Target(
        start=4700,
        end=7500,
        threshold=0.02,
        vmicro=2.9,
        teff=7550,
        logg=4.08,
        mh=+0.22,
        comment="atmo-retrieval PySME line list; TOI-1431/MASCARA-5; vacuum; 4700-7500 A",
    ),
    "wasp33b": Target(
        start=4700,
        end=7500,
        threshold=0.02,
        vmicro=3.8,
        teff=7460,
        logg=4.30,
        mh=-0.04,
        comment="atmo-retrieval PySME line list; WASP-33; vacuum; 4700-7500 A",
    ),
    "wasp189b": Target(
        start=4200,
        end=7500,
        threshold=0.02,
        vmicro=3.3,
        teff=8220,
        logg=4.03,
        mh=+0.15,
        comment="atmo-retrieval PySME line list; WASP-189; vacuum; 4200-7500 A",
    ),
    "mascara1b": Target(
        start=4700,
        end=7500,
        threshold=0.02,
        vmicro=2.9,
        teff=7700,
        logg=4.11,
        mh=-0.01,
        comment="atmo-retrieval PySME line list; MASCARA-1; vacuum; 4700-7500 A",
    ),
    "hatp11b_red": Target(
        start=6200,
        end=7500,
        threshold=0.02,
        vmicro=1.0,
        teff=4780,
        logg=4.59,
        mh=+0.31,
        comment="atmo-retrieval PySME line list; HAT-P-11; vacuum; 6200-7500 A; red arm only",
    ),
    "hatp11b_blue": Target(
        start=4700,
        end=5500,
        threshold=0.02,
        vmicro=1.0,
        teff=4780,
        logg=4.59,
        mh=+0.31,
        comment="atmo-retrieval PySME line list; HAT-P-11; vacuum; 4700-5500 A; blue arm only",
    ),
    "kelt5b": Target(
        start=4200,
        end=6350,
        threshold=0.02,
        vmicro=2.0,
        teff=7128,
        logg=4.24,
        mh=0.00,
        comment="atmo-retrieval PySME line list; KELT-5b; vacuum; 4200-6350 A",
    ),
    "toi1789b": Target(
        start=6200,
        end=7500,
        threshold=0.02,
        vmicro=1.0,
        teff=5984,
        logg=3.94,
        mh=+0.37,
        comment="atmo-retrieval PySME line list; TOI-1789; vacuum; 6200-7500 A; red arm only",
    ),
    "v1298taub": Target(
        start=4700,
        end=7500,
        threshold=0.02,
        vmicro=0.85,
        teff=5050,
        logg=4.25,
        mh=+0.10,
        comment="atmo-retrieval PySME line list; V1298 Tau; vacuum; 4700-7500 A",
    ),
}


def _wait(driver) -> WebDriverWait:
    return WebDriverWait(driver, WAIT_SECONDS)


def _set_control(control: WebElement, value: object) -> None:
    control.clear()
    control.send_keys(str(value))


def _visible_text_controls(driver) -> list[WebElement]:
    controls = driver.find_elements(
        By.XPATH,
        "//input[not(@type='hidden') and "
        "(@type='text' or @type='number' or not(@type))] | //textarea",
    )
    return [control for control in controls if control.is_displayed()]


def _describe_controls(driver) -> str:
    entries = []
    for control in _visible_text_controls(driver):
        entries.append(
            f"<{control.tag_name} name={control.get_attribute('name')!r} "
            f"type={control.get_attribute('type')!r}>"
        )
    return ", ".join(entries)


def _find_text_control(
    driver,
    *,
    names: tuple[str, ...],
    label_fragments: tuple[str, ...],
) -> WebElement:
    """
    Find a VALD text field.

    Strategy:
    1. Try exact known HTML name attributes.
    2. Find the specific label/table cell containing the requested text and
       take the text control immediately associated with / beside that label.

    Do NOT search an entire <tr> and return its first text input: VALD's form
    layout can place multiple controls in a broad row/container, which can
    cause e.g. "Detection limit" to resolve to the composition textarea.
    """
    # 1) Exact HTML name match. Refuse ambiguity instead of silently selecting
    # whichever matching control Selenium happens to return first.
    for name in names:
        elements = [
            element
            for element in driver.find_elements(By.NAME, name)
            if element.is_displayed()
        ]
        if len(elements) == 1:
            return elements[0]
        if len(elements) > 1:
            raise RuntimeError(
                f"VALD has multiple visible text controls named {name!r}."
            )

    # 2) Label/cell-local match.
    fragments = tuple(
        " ".join(fragment.lower().split()) for fragment in label_fragments
    )

    # Search only label-like elements, not arbitrary large containers.
    candidates = driver.find_elements(By.XPATH, "//label | //td | //th")
    for label in candidates:
        if not label.is_displayed():
            continue

        # Selenium's .text includes every descendant. VALD's malformed legacy
        # table markup creates page-wide <td> elements whose descendant text
        # contains every form label; matching those is what caused several
        # logical fields to alias the first input on the page. Match only text
        # nodes belonging directly to this label/cell.
        direct_text = driver.execute_script(
            "return Array.from(arguments[0].childNodes)"
            ".filter(node => node.nodeType === 3)"
            ".map(node => node.textContent).join(' ');",
            label,
        )
        label_text = " ".join((direct_text or "").lower().split())
        if not any(fragment in label_text for fragment in fragments):
            continue

        # Standard <label for="..."> association.
        for_attr = label.get_attribute("for")
        if for_attr:
            associated = driver.find_elements(By.ID, for_attr)
            for element in associated:
                if element.is_displayed():
                    return element

        # Input/textarea directly inside the label/cell. Do not use .// here:
        # it would search all descendants of a page-wide table cell.
        nested = label.find_elements(
            By.XPATH,
            "./input[not(@type='hidden') and "
            "(@type='text' or @type='number' or not(@type))] | ./textarea",
        )
        for element in nested:
            if element.is_displayed():
                return element

        # Typical VALD table layout:
        #   <td>Detection limit:</td><td><input ...></td>
        sibling_controls = label.find_elements(
            By.XPATH,
            "./following-sibling::*[1]"
            "/input[not(@type='hidden') and "
            "(@type='text' or @type='number' or not(@type))] | "
            "./following-sibling::*[1]/textarea",
        )
        for element in sibling_controls:
            if element.is_displayed():
                return element

        # Some simple forms put the control as a direct sibling.
        direct_sibling = label.find_elements(
            By.XPATH,
            "./following-sibling::input[1] | ./following-sibling::textarea[1]",
        )
        for element in direct_sibling:
            if element.is_displayed():
                return element

    raise RuntimeError(
        "Could not locate VALD field for "
        f"{label_fragments}. Visible text controls: {_describe_controls(driver)}"
    )


def _assert_numeric_control(
    control: WebElement,
    expected: float,
    field_name: str,
) -> None:
    raw_value = (control.get_attribute("value") or "").strip()
    try:
        actual = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"VALD {field_name} was not populated correctly. "
            f"Expected {expected}, found {raw_value!r}."
        ) from exc

    if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"VALD {field_name} was not populated correctly. "
            f"Expected {expected}, found {raw_value!r}."
        )


def _click_input_value(driver, *values: str) -> WebElement:
    for value in values:
        xpath = f"//input[translate(@value,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')=" \
                f"'{value.lower()}']"
        elements = driver.find_elements(By.XPATH, xpath)
        for element in elements:
            if element.is_displayed():
                if not element.is_selected():
                    element.click()
                return element
    raise RuntimeError(f"Could not find VALD input with value in {values!r}")


def _ensure_checkbox_off_by_value(driver, *values: str) -> None:
    for value in values:
        xpath = (
            "//input[@type='checkbox' and "
            "translate(@value,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')="
            f"'{value.lower()}']"
        )
        for element in driver.find_elements(By.XPATH, xpath):
            if element.is_displayed() and element.is_selected():
                element.click()


def _ensure_checkbox_off_by_row(driver, *fragments: str) -> None:
    fragments_lower = tuple(fragment.lower() for fragment in fragments)
    for row in driver.find_elements(By.XPATH, "//tr"):
        text = " ".join(row.text.lower().split())
        if not any(fragment in text for fragment in fragments_lower):
            continue
        for element in row.find_elements(By.XPATH, ".//input[@type='checkbox']"):
            if element.is_displayed() and element.is_selected():
                element.click()


def _validate_email(email: str) -> str:
    """Minimal sanity check for the registered VALD email address."""
    email = email.strip()
    local, sep, domain = email.rpartition("@")
    if not local or sep != "@" or not domain or "." not in domain:
        raise SystemExit(f"Invalid email address: {email!r}")
    return email


def login(driver, email: str) -> None:
    driver.get(VALD_URL)

    login_box = _wait(driver).until(EC.presence_of_element_located((By.NAME, "user")))
    login_box.clear()
    login_box.send_keys(email)

    submit = driver.find_element(By.XPATH, "//input[@type='submit']")
    submit.click()

    # The unauthenticated page itself also says "Welcome to VALD3", so that
    # phrase cannot be used as evidence of a successful login. Require an
    # authenticated control or the explicit "Logged in as" text instead.
    try:
        _wait(driver).until(
            lambda d: (
                len(d.find_elements(By.XPATH, "//input[@value='Extract Stellar']")) > 0
                or "logged in as" in d.find_element(By.TAG_NAME, "body").text.lower()
            )
        )
    except TimeoutException as exc:
        body = driver.find_element(By.TAG_NAME, "body").text
        raise RuntimeError(
            "VALD login did not succeed. Check that --email is exactly the "
            "registered VALD email address. Current page begins:\n"
            + body[:500]
        ) from exc


def open_extract_stellar(driver) -> None:
    driver.get(VALD_URL)
    button = _wait(driver).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@value='Extract Stellar']"))
    )
    button.click()
    _wait(driver).until(EC.presence_of_element_located((By.NAME, "stwvl")))



def _normalized(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _click_control_with_text(driver, phrases: tuple[str, ...]) -> WebElement:
    """Click a visible link/button/input whose text or value contains a phrase."""
    phrases = tuple(_normalized(p) for p in phrases)

    candidates = driver.find_elements(
        By.XPATH,
        "//a | //button | //input",
    )
    for element in candidates:
        if not element.is_displayed():
            continue
        haystack = _normalized(
            " ".join(
                part
                for part in (
                    element.text,
                    element.get_attribute("value"),
                    element.get_attribute("title"),
                    element.get_attribute("alt"),
                )
                if part
            )
        )
        if any(phrase in haystack for phrase in phrases):
            element.click()
            return element

    raise RuntimeError(f"Could not find visible VALD control matching {phrases!r}")


def _choose_option_in_container(
    container: WebElement,
    desired_phrases: tuple[str, ...],
) -> bool:
    """Choose a select/radio option inside one VALD unit-setting container."""
    desired = tuple(_normalized(p) for p in desired_phrases)

    # Handle ordinary <select> menus.
    for select_el in container.find_elements(By.TAG_NAME, "select"):
        if not select_el.is_displayed():
            continue
        selector = Select(select_el)
        for option in selector.options:
            label = _normalized(option.text)
            value = _normalized(option.get_attribute("value"))
            if any(p in label or p in value for p in desired):
                selector.select_by_value(option.get_attribute("value"))
                return True

    # Handle radio buttons / checkboxes.
    for control in container.find_elements(
        By.XPATH, ".//input[@type='radio' or @type='checkbox']"
    ):
        if not control.is_displayed():
            continue

        parts = [
            control.get_attribute("value") or "",
            control.get_attribute("title") or "",
        ]

        control_id = control.get_attribute("id")
        if control_id:
            labels = container.find_elements(
                By.XPATH, f".//label[@for={control_id!r}]"
            )
            parts.extend(label.text for label in labels)

        # VALD forms are table-like; parent text often contains the option label.
        try:
            parts.append(control.find_element(By.XPATH, "./..").text)
        except Exception:
            pass

        haystack = _normalized(" ".join(parts))
        if any(p in haystack for p in desired):
            if not control.is_selected():
                control.click()
            return True

    return False


def _set_unit_setting(
    driver,
    *,
    setting_phrases: tuple[str, ...],
    desired_phrases: tuple[str, ...],
) -> None:
    """Set one unit preference by finding the row/container with its label."""
    setting = tuple(_normalized(p) for p in setting_phrases)

    # VALD currently uses a compact table form. Search rows first.
    containers = driver.find_elements(By.XPATH, "//tr")
    # Fallbacks make this tolerant of modest HTML layout changes.
    containers += driver.find_elements(By.XPATH, "//fieldset")
    containers += driver.find_elements(By.XPATH, "//div")

    seen = set()
    for container in containers:
        key = container.id
        if key in seen or not container.is_displayed():
            continue
        seen.add(key)

        text = _normalized(container.text)
        if not any(p in text for p in setting):
            continue

        if _choose_option_in_container(container, desired_phrases):
            return

    raise RuntimeError(
        f"Could not set VALD unit preference {setting_phrases!r} "
        f"to {desired_phrases!r}."
    )


def configure_vald_units(driver) -> None:
    """Set and save the unit preferences required by the PySME line-list requests."""
    # "Unit selection" is exposed on the extraction form, not the VALD home page.
    open_extract_stellar(driver)

    _click_control_with_text(driver, ("unit selection", "unit selections"))

    _wait(driver).until(
        lambda d: "medium" in _normalized(d.find_element(By.TAG_NAME, "body").text)
    )

    _set_unit_setting(
        driver,
        setting_phrases=("energy unit", "energy"),
        desired_phrases=("ev",),
    )
    _set_unit_setting(
        driver,
        setting_phrases=("medium",),
        desired_phrases=("vacuum",),
    )
    _set_unit_setting(
        driver,
        setting_phrases=("wavelength unit", "wave unit"),
        desired_phrases=("angstrom", "ångstrom"),
    )
    _set_unit_setting(
        driver,
        setting_phrases=("vdw syntax", "van der waals syntax"),
        desired_phrases=("default",),
    )

    _click_control_with_text(driver, ("save settings", "save"))

    # Re-open Extract Stellar and verify the saved summary before any request.
    open_extract_stellar(driver)
    verify_saved_units(driver)
    print("VALD units configured: eV / vacuum / angstrom / default")


def verify_saved_units(driver) -> None:
    text = " ".join(driver.find_element(By.TAG_NAME, "body").text.lower().split())

    required = {
        "vacuum": "medium: vacuum",
        "angstrom": "wavelength unit: angstrom",
        "eV": "energy unit: ev",
    }
    missing = [label for label, phrase in required.items() if phrase not in text]

    if missing:
        raise RuntimeError(
            "VALD saved Unit selection is not the configuration required by this "
            f"script (missing: {', '.join(missing)}).\n"
            "In VALD, open Unit selection and save:\n"
            "  Energy unit: eV\n"
            "  Medium: vacuum\n"
            "  Wavelength unit: angstrom\n"
            "  VdW syntax: default\n"
            "Then rerun this script."
        )


def fill_extract_stellar(driver, target: Target) -> None:
    # Set radio/checkbox options FIRST. If VALD changes form state when these are
    # clicked, the required numerical fields below are filled only afterward.
    _click_input_value(driver, "short")
    _click_input_value(driver, "via ftp")
    _click_input_value(driver, "default")

    _ensure_checkbox_off_by_value(driver, "HFS splitting")
    _ensure_checkbox_off_by_row(driver, "radiative damping")
    _ensure_checkbox_off_by_row(driver, "stark damping")
    _ensure_checkbox_off_by_row(driver, "van der waals")
    _ensure_checkbox_off_by_row(driver, "landé factor", "lande factor")
    _ensure_checkbox_off_by_row(driver, "term designation")

    fields = {
        "start": _find_text_control(
            driver,
            names=("stwvl",),
            label_fragments=("starting wavelength",),
        ),
        "end": _find_text_control(
            driver,
            names=("endwvl",),
            label_fragments=("ending wavelength",),
        ),
        "threshold": _find_text_control(
            driver,
            names=("dlimit", "detlim", "threshold", "depth", "detect"),
            label_fragments=("detection limit", "detection threshold"),
        ),
        "vmicro": _find_text_control(
            driver,
            names=("micturb", "vmicro", "vmic", "micro"),
            label_fragments=("microturbulence",),
        ),
        "teff": _find_text_control(
            driver,
            names=("teff",),
            label_fragments=("teff",),
        ),
        "logg": _find_text_control(
            driver,
            names=("logg", "log_g"),
            label_fragments=("log g",),
        ),
        "composition": _find_text_control(
            driver,
            names=("chemcomp", "abund", "abundance", "composition", "chem"),
            label_fragments=("chemical composition",),
        ),
    }

    # A selector fallback must never assign one DOM control to multiple logical
    # fields. Catch that before any values are entered and before submission.
    controls_by_id: dict[str, list[str]] = {}
    for field_name, control in fields.items():
        controls_by_id.setdefault(control.id, []).append(field_name)
    aliases = [names for names in controls_by_id.values() if len(names) > 1]
    if aliases:
        raise RuntimeError(f"VALD field selectors alias the same control: {aliases}")

    _set_control(fields["start"], target.start)
    _set_control(fields["end"], target.end)
    _set_control(fields["threshold"], target.threshold)
    _set_control(fields["vmicro"], target.vmicro)
    _set_control(fields["teff"], target.teff)
    _set_control(fields["logg"], target.logg)
    _set_control(fields["composition"], f"M/H: {target.mh:+.2f}")

    # Optional comment. HELIOS-K identifies this field as "subject" in the
    # VALD forms; keep a label-based fallback in case that changes.
    try:
        comment = _find_text_control(
            driver,
            names=("subject",),
            label_fragments=("optional comment",),
        )
        _set_control(comment, target.comment)
    except RuntimeError:
        print("  warning: could not locate optional comment field", file=sys.stderr)

    # Validate every required field before pressing Submit. Checking only the
    # detection threshold previously left other selector mistakes invisible.
    numeric_values = {
        "starting wavelength": (fields["start"], target.start),
        "ending wavelength": (fields["end"], target.end),
        "detection threshold": (fields["threshold"], target.threshold),
        "microturbulence": (fields["vmicro"], target.vmicro),
        "Teff": (fields["teff"], target.teff),
        "log g": (fields["logg"], target.logg),
    }
    for field_name, (control, expected) in numeric_values.items():
        _assert_numeric_control(control, expected, field_name)

    expected_composition = f"M/H: {target.mh:+.2f}"
    actual_composition = (fields["composition"].get_attribute("value") or "").strip()
    if actual_composition != expected_composition:
        raise RuntimeError(
            "VALD chemical composition was not populated correctly. "
            f"Expected {expected_composition!r}, found {actual_composition!r}."
        )


def submit_current_request(driver) -> None:
    submit = _wait(driver).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@value='Submit request']"))
    )
    submit.click()

    # A successful form POST replaces the current document, making this exact
    # button stale even if the response happens to contain another Submit
    # control. If client-side validation blocks submission, it stays attached.
    try:
        _wait(driver).until(EC.staleness_of(submit))
    except TimeoutException as exc:
        body = driver.find_element(By.TAG_NAME, "body").text
        raise RuntimeError(
            "VALD did not navigate after Submit request; browser-side form "
            "validation probably rejected the request. Current page begins:\n"
            + body[:800]
        ) from exc


def submit_target(driver, name: str, target: Target) -> None:
    print(
        f"[{name}] {target.start:.0f}-{target.end:.0f} A, "
        f"Teff={target.teff}, logg={target.logg:.2f}, "
        f"[M/H]={target.mh:+.2f}, vmicro={target.vmicro:.2f}"
    )

    open_extract_stellar(driver)
    verify_saved_units(driver)
    fill_extract_stellar(driver, target)
    submit_current_request(driver)
    print(f"[{name}] submitted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit VALD3 Extract Stellar requests for PySME line lists."
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("VALD_EMAIL"),
        help="Registered VALD email address. Can also set VALD_EMAIL.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(TARGETS),
        help="Submit only these target keys. Default: all targets.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Firefox headlessly. Visible Firefox is recommended for the first run.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation before submitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.email:
        raise SystemExit(
            "Provide your registered VALD email with --email or set VALD_EMAIL."
        )
    args.email = _validate_email(args.email)

    selected = args.only if args.only else list(TARGETS)
    print("Targets to submit:")
    for name in selected:
        print(f"  - {name}")

    if not args.yes:
        answer = input(f"\nSubmit {len(selected)} VALD3 request(s)? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Nothing submitted.")

    options = FirefoxOptions()
    if args.headless:
        options.add_argument("-headless")

    driver = webdriver.Firefox(options=options)

    try:
        login(driver, args.email)
        configure_vald_units(driver)

        for index, name in enumerate(selected):
            submit_target(driver, name, TARGETS[name])
            if index < len(selected) - 1:
                time.sleep(BETWEEN_REQUESTS_SECONDS)

    finally:
        driver.quit()

    print(f"Done. Submitted {len(selected)} request(s).")


if __name__ == "__main__":
    main()
