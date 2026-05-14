using System.Globalization;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

public class Quest3UdpSender : MonoBehaviour
{
    public string pcIp = "192.168.1.100";
    public int pcPort = 5055;
    public float sendHz = 60.0f;

    public bool showIpPanelOnStart = true;

    private const string SavedIpKey = "quest3_udp_sender_pc_ip";
    private const int KeyboardButtonCount = 15;

    private UdpClient udp;
    private InputDevice rightHand;
    private float timer;
    private bool previousPrimaryButton;
    private bool previousSecondaryButton;
    private bool previousTriggerButton;
    private bool previousGripButton;
    private bool previousStickClick;
    private float secondaryHoldSeconds;

    private GameObject ipPanelRoot;
    private TextMesh ipPanelText;
    private TextMesh ipPanelShadowText;
    private bool ipPanelOpen;
    private TextMesh[] keyboardButtons = new TextMesh[KeyboardButtonCount];
    private string[] keyboardValues = new string[KeyboardButtonCount];
    private Vector3[] keyboardLocalPositions = new Vector3[KeyboardButtonCount];
    private int selectedKeyIndex;
    private float nextStickEditTime;
    private string keyboardText = "";
    private string inputError = "";

    void Start()
    {
        pcIp = PlayerPrefs.GetString(SavedIpKey, pcIp);
        keyboardText = pcIp;
        ipPanelOpen = showIpPanelOnStart;

        udp = new UdpClient();
        rightHand = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);

        CreateIpPanel();
        RefreshIpPanel();
    }

    void Update()
    {
        if (!rightHand.isValid)
            rightHand = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);

        rightHand.TryGetFeatureValue(CommonUsages.primaryButton, out bool calibrate);
        rightHand.TryGetFeatureValue(CommonUsages.secondaryButton, out bool resetCalibration);
        rightHand.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerButton);
        rightHand.TryGetFeatureValue(CommonUsages.gripButton, out bool gripButton);
        rightHand.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);
        rightHand.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out bool stickClick);

        if (!ipPanelOpen)
        {
            if (resetCalibration)
                secondaryHoldSeconds += Time.deltaTime;
            else
                secondaryHoldSeconds = 0.0f;

            if ((stickClick && !previousStickClick) || secondaryHoldSeconds > 1.2f)
            {
                OpenIpPanel();
            }
        }

        if (ipPanelOpen)
        {
            UpdateSelectedKeyFromController();
            HandleIpPanelInput(calibrate, triggerButton, gripButton, resetCalibration, stickClick, stick);
            previousPrimaryButton = calibrate;
            previousSecondaryButton = resetCalibration;
            previousTriggerButton = triggerButton;
            previousGripButton = gripButton;
            previousStickClick = stickClick;
            return;
        }

        timer += Time.deltaTime;
        if (timer < 1.0f / sendHz)
        {
            previousPrimaryButton = calibrate;
            previousSecondaryButton = resetCalibration;
            previousTriggerButton = triggerButton;
            previousGripButton = gripButton;
            previousStickClick = stickClick;
            return;
        }
        timer = 0.0f;

        rightHand.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 pos);
        rightHand.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rot);
        rightHand.TryGetFeatureValue(CommonUsages.gripButton, out bool grip);
        rightHand.TryGetFeatureValue(CommonUsages.triggerButton, out bool trigger);

        Vector3 p = pos;
        Quaternion q = rot;

        string json = string.Format(
            CultureInfo.InvariantCulture,
            "{{\"pose\":{{\"position\":[{0:F6},{1:F6},{2:F6}],\"orientation\":[{3:F6},{4:F6},{5:F6},{6:F6}]}},\"grip_pressed\":{7},\"trigger_pressed\":{8},\"calibrate_pressed\":{9},\"reset_calibration_pressed\":{10}}}",
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
            grip.ToString().ToLowerInvariant(),
            trigger.ToString().ToLowerInvariant(),
            calibrate.ToString().ToLowerInvariant(),
            resetCalibration.ToString().ToLowerInvariant()
        );

        byte[] data = Encoding.UTF8.GetBytes(json);
        udp.Send(data, data.Length, pcIp, pcPort);

        previousPrimaryButton = calibrate;
        previousSecondaryButton = resetCalibration;
        previousTriggerButton = triggerButton;
        previousGripButton = gripButton;
        previousStickClick = stickClick;
    }

    void OnDestroy()
    {
        udp?.Close();
    }

    private void HandleIpPanelInput(
        bool primaryButton,
        bool triggerButton,
        bool gripButton,
        bool secondaryButton,
        bool stickClick,
        Vector2 stick
    )
    {
        if (
            (primaryButton && !previousPrimaryButton) ||
            (triggerButton && !previousTriggerButton) ||
            (gripButton && !previousGripButton) ||
            (stickClick && !previousStickClick)
        )
        {
            PressSelectedKey();
            return;
        }

        if (secondaryButton && !previousSecondaryButton)
        {
            CloseIpPanelWithoutSaving();
            return;
        }

        HandleKeyboardNavigation(stick);
    }

    private void OpenIpPanel()
    {
        ipPanelOpen = true;
        secondaryHoldSeconds = 0.0f;
        keyboardText = pcIp;
        inputError = "";
        RefreshIpPanel();
    }

    private void HandleKeyboardNavigation(Vector2 stick)
    {
        UpdateSelectedKeyFromController();

        if (Time.time < nextStickEditTime)
            return;

        int row = selectedKeyIndex / 3;
        int col = selectedKeyIndex % 3;
        bool moved = false;

        if (stick.x > 0.65f)
        {
            col = Mathf.Min(2, col + 1);
            moved = true;
        }
        else if (stick.x < -0.65f)
        {
            col = Mathf.Max(0, col - 1);
            moved = true;
        }
        else if (stick.y > 0.65f)
        {
            row = Mathf.Max(0, row - 1);
            moved = true;
        }
        else if (stick.y < -0.65f)
        {
            row = Mathf.Min(4, row + 1);
            moved = true;
        }

        if (!moved)
            return;

        selectedKeyIndex = Mathf.Clamp(row * 3 + col, 0, KeyboardButtonCount - 1);
        nextStickEditTime = Time.time + 0.18f;
        RefreshIpPanel();
    }

    private void PressSelectedKey()
    {
        string value = keyboardValues[selectedKeyIndex];
        inputError = "";

        if (value == "DEL")
        {
            if (keyboardText.Length > 0)
                keyboardText = keyboardText.Substring(0, keyboardText.Length - 1);
            RefreshIpPanel();
            return;
        }

        if (value == "CLR")
        {
            keyboardText = "";
            RefreshIpPanel();
            return;
        }

        if (value == "OK")
        {
            TrySaveIpFromText();
            return;
        }

        if (value == "CLOSE")
        {
            CloseIpPanelWithoutSaving();
            return;
        }

        if (keyboardText.Length < 15)
            keyboardText += value;
        RefreshIpPanel();
    }

    private void TrySaveIpFromText()
    {
        string candidate = keyboardText.Trim();

        if (!IsValidIpv4(candidate))
        {
            inputError = "Invalid IP: " + candidate;
            keyboardText = candidate;
            RefreshIpPanel();
            return;
        }

        pcIp = candidate;
        PlayerPrefs.SetString(SavedIpKey, pcIp);
        PlayerPrefs.Save();
        ipPanelOpen = false;
        inputError = "";
        RefreshIpPanel();
    }

    private void CloseIpPanelWithoutSaving()
    {
        keyboardText = pcIp;
        inputError = "";
        ipPanelOpen = false;
        secondaryHoldSeconds = 0.0f;
        RefreshIpPanel();
    }

    private bool IsValidIpv4(string ip)
    {
        string[] parts = ip.Split('.');
        if (parts.Length != 4)
            return false;

        for (int i = 0; i < parts.Length; i++)
        {
            if (
                !int.TryParse(parts[i], NumberStyles.None, CultureInfo.InvariantCulture, out int value) ||
                value < 0 ||
                value > 255
            )
            {
                return false;
            }
        }
        return true;
    }

    private void UpdateSelectedKeyFromController()
    {
        if (!rightHand.isValid || ipPanelRoot == null)
            return;

        if (
            !rightHand.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 controllerPosition) ||
            !rightHand.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion controllerRotation)
        )
        {
            return;
        }

        Vector3 rayOrigin = controllerPosition;
        Vector3 rayDirection = controllerRotation * Vector3.forward;
        Plane panelPlane = new Plane(ipPanelRoot.transform.forward, ipPanelRoot.transform.position);
        if (!panelPlane.Raycast(new Ray(rayOrigin, rayDirection), out float distance))
            return;
        if (distance < 0.0f || distance > 4.0f)
            return;

        Vector3 localHit = ipPanelRoot.transform.InverseTransformPoint(rayOrigin + rayDirection * distance);
        int bestIndex = selectedKeyIndex;
        float bestDistance = 0.08f;
        for (int i = 0; i < KeyboardButtonCount; i++)
        {
            float d = Vector2.Distance(
                new Vector2(localHit.x, localHit.y),
                new Vector2(keyboardLocalPositions[i].x, keyboardLocalPositions[i].y)
            );
            if (d < bestDistance)
            {
                bestDistance = d;
                bestIndex = i;
            }
        }

        if (bestIndex != selectedKeyIndex)
        {
            selectedKeyIndex = bestIndex;
            RefreshIpPanel();
        }
    }

    private void CreateIpPanel()
    {
        Transform parent = Camera.main != null ? Camera.main.transform : null;

        ipPanelRoot = new GameObject("Quest3IpPanel");
        if (parent != null)
        {
            ipPanelRoot.transform.SetParent(parent, false);
            ipPanelRoot.transform.localPosition = new Vector3(0.0f, -0.04f, 1.25f);
            ipPanelRoot.transform.localRotation = Quaternion.identity;
        }
        else
        {
            ipPanelRoot.transform.position = new Vector3(0.0f, 1.5f, 1.5f);
        }

        GameObject shadowTextObject = new GameObject("ShadowText");
        shadowTextObject.transform.SetParent(ipPanelRoot.transform, false);
        shadowTextObject.transform.localPosition = new Vector3(0.006f, 0.394f, 0.01f);
        ipPanelShadowText = shadowTextObject.AddComponent<TextMesh>();
        ipPanelShadowText.anchor = TextAnchor.MiddleCenter;
        ipPanelShadowText.alignment = TextAlignment.Center;
        ipPanelShadowText.fontSize = 56;
        ipPanelShadowText.characterSize = 0.014f;
        ipPanelShadowText.color = Color.black;

        GameObject textObject = new GameObject("Text");
        textObject.transform.SetParent(ipPanelRoot.transform, false);
        textObject.transform.localPosition = new Vector3(0.0f, 0.4f, 0.0f);
        ipPanelText = textObject.AddComponent<TextMesh>();
        ipPanelText.anchor = TextAnchor.MiddleCenter;
        ipPanelText.alignment = TextAlignment.Center;
        ipPanelText.fontSize = 56;
        ipPanelText.characterSize = 0.014f;
        ipPanelText.color = new Color(0.0f, 1.0f, 0.85f, 1.0f);

        CreateKeyboardButtons();
    }

    private void CreateKeyboardButtons()
    {
        string[] values = {
            "1", "2", "3",
            "4", "5", "6",
            "7", "8", "9",
            ".", "0", "DEL",
            "CLR", "OK", "CLOSE"
        };

        for (int i = 0; i < KeyboardButtonCount; i++)
        {
            keyboardValues[i] = values[i];
            int row = i / 3;
            int col = i % 3;
            keyboardLocalPositions[i] = new Vector3((col - 1) * 0.28f, 0.08f - row * 0.105f, 0.0f);
            GameObject keyObject = new GameObject("Key_" + values[i]);
            keyObject.transform.SetParent(ipPanelRoot.transform, false);
            keyObject.transform.localPosition = keyboardLocalPositions[i];
            TextMesh textMesh = keyObject.AddComponent<TextMesh>();
            textMesh.anchor = TextAnchor.MiddleCenter;
            textMesh.alignment = TextAlignment.Center;
            textMesh.fontSize = 72;
            textMesh.characterSize = 0.014f;
            keyboardButtons[i] = textMesh;
        }
    }

    private void RefreshIpPanel()
    {
        if (ipPanelRoot == null || ipPanelText == null || ipPanelShadowText == null)
            return;

        ipPanelRoot.SetActive(ipPanelOpen);
        if (!ipPanelOpen)
            return;

        string text =
            "PC IP\n" +
            keyboardText + ":" + pcPort + "\n" +
            inputError + (string.IsNullOrEmpty(inputError) ? "" : "\n\n") +
            "Point key + Trigger/Grip/A\n" +
            "Stick also selects keys\n" +
            "B: close";
        ipPanelText.text = text;
        ipPanelShadowText.text = text;

        for (int i = 0; i < KeyboardButtonCount; i++)
        {
            if (keyboardButtons[i] == null)
                continue;
            string value = keyboardValues[i];
            bool selected = i == selectedKeyIndex;
            keyboardButtons[i].text = selected ? "[" + value + "]" : value;
            keyboardButtons[i].color = selected ? Color.yellow : Color.white;
            keyboardButtons[i].gameObject.SetActive(ipPanelOpen);
        }
    }
}
