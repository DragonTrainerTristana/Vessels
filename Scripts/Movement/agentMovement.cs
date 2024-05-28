using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.IO;

public class AgentMovement : Agent
{
    // Target
    public Transform Goal;
    public Transform MyPos;
    public float goalDistance;
    private float preDistance;
    public float rdVelocity;

    // Vessel Physcial parameter
    public float maxPower = 0.05f;
    public float maxRudderAngle = 0.05f;
    public float maxSpeed = 2f; // 최대 속도 정의

    // RayCast & LineRenderer
    public float rayLength;
    public LineRenderer lineRenderer;
    public int rayCount;
    public float raySpreadAngle;
    public float[] rayObjDis;

    // Unity Basic Component (Essential)
    private Rigidbody rd;

    private string logFilePath;

    // Additional Check
    public float powerAction;
    public float rudderAction;

    public override void Initialize()
    {
        // Get Basic Component 
        rd = GetComponent<Rigidbody>();

        lineRenderer = gameObject.AddComponent<LineRenderer>(); // 라인 렌더러 컴포넌트 추가
        lineRenderer.positionCount = rayCount * 2; // 레이의 시작점과 끝점
        lineRenderer.startWidth = 0.01f; // 라인의 시작 폭
        lineRenderer.endWidth = 0.01f; // 라인의 끝 폭
        lineRenderer.material = new Material(Shader.Find("Sprites/Default")); // 라인의 재질
        lineRenderer.startColor = Color.red; // 라인의 시작 색
        lineRenderer.endColor = Color.red; // 라인의 끝 색

        // Hyperparameter Vessel
        maxPower = 0.05f; // 최대 전진 속도
        maxRudderAngle = 0.3f; // 최대 러더 각도

        // Hyperparameter RayCast 
        rayLength = 10.0f;
        rayCount = 360;
        raySpreadAngle = 1f;

        MyPos = this.gameObject.transform;

        // 로그 파일 경로 설정
        logFilePath = Path.Combine(Application.persistentDataPath, "log.txt");
        Application.logMessageReceived += HandleLog;
    }

    private void OnDestroy()
    {
        Application.logMessageReceived -= HandleLog;
    }

    private void HandleLog(string logString, string stackTrace, LogType type)
    {
        File.AppendAllText(logFilePath, logString + "\n");
    }

    public override void OnEpisodeBegin()
    {
        // My Position
        this.gameObject.transform.position = MyPos.position;
        rd.velocity = Vector3.zero; // 에피소드 시작 시 속도 초기화
        rd.angularVelocity = Vector3.zero; // 에피소드 시작 시 각속도 초기화

        goalDistance = Vector3.Distance(Goal.position, this.gameObject.transform.position);
        preDistance = goalDistance;

        // RayCast & LineRenderer 매 Episode마다 호출
        rayObjDis = new float[rayCount]; // 각 레이의 거리 정보 배열 초기화
        for (int i = 0; i < rayCount; i++) rayObjDis[i] = rayLength; // 초기 거리를 최대 레이 길이로 설정
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        powerAction = actions.ContinuousActions[0]; // Power
        rudderAction = actions.ContinuousActions[1]; // Rudder

        // 디버그 로그 추가    
        // 근데 할 필요가 없음...
        // 따로 출력해야 함.

        Debug.Log($"Power Action: {powerAction}, Rudder Action: {rudderAction}");

        agentMovement(powerAction, rudderAction);

        // Find fastest way
        AddReward(-0.0001f);
        // Reward Setting
        if (goalDistance < 1.0f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (goalDistance > 50.0f)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
        else
        {
            // Reward 세팅
            preDistance = Vector3.Distance(Goal.position, this.gameObject.transform.position);

            if (preDistance < goalDistance)
            {
                AddReward(-0.001f);
            }
            else
            {
                AddReward(0.001f);
                goalDistance = preDistance;
            }
        }
    }

    public void agentMovement(float powerAction, float rudderAction)
    {
        float power = maxPower * Mathf.Clamp(powerAction, -1f, 1f);
        float rudderAngle = maxRudderAngle * Mathf.Clamp(rudderAction, -1f, 1f);

        if (rd.velocity.magnitude > maxSpeed)
        {
            rd.velocity = rd.velocity.normalized * maxSpeed;
        }

        Quaternion turnRotation = Quaternion.Euler(0f, rudderAngle, 0f);
        rd.MoveRotation(rd.rotation * turnRotation);

        Vector3 force = transform.right * power;
        rdVelocity = force.z;
        rd.AddForce(force, ForceMode.VelocityChange);

        Debug.Log($"Applied Force: {force}, Current Velocity: {rd.velocity}");
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * raySpreadAngle - 180; // 각도를 계산하여 360도 전체에 레이를 균등하게 분포
            Quaternion rotation = Quaternion.Euler(0f, angle, 0f); // 회전값 설정
            Vector3 direction = rotation * transform.forward; // 방향 벡터 계산

            Ray ray = new Ray(transform.position, direction); // 레이 생성
            RaycastHit hit; // 레이캐스트 히트 정보

            int index = i * 2; // 라인 렌더러의 위치 인덱스 계산
            lineRenderer.SetPosition(index, ray.origin); // 라인의 시작점 설정

            if (Physics.Raycast(ray, out hit, rayLength))
            {
                // 레이가 객체에 충돌한 경우
                lineRenderer.SetPosition(index + 1, hit.point); // 라인의 끝점을 충돌 지점으로 설정
                rayObjDis[i] = hit.distance; // 충돌 거리 저장
            }
            else
            {
                // 레이가 충돌하지 않은 경우
                lineRenderer.SetPosition(index + 1, ray.origin + ray.direction * rayLength); // 라인의 끝점을 최대 길이로 설정
                rayObjDis[i] = rayLength; // 최대 거리로 설정
            }
        }

        // RayCast 360개 Obs
        sensor.AddObservation(rayObjDis);

        // My Position - 3개 <Message Actor용>
        sensor.AddObservation(this.gameObject.transform.position);

        // Distance from Goal - 1개
        sensor.AddObservation(Vector3.Distance(Goal.position, this.gameObject.transform.position));
    }


    void OnCollisionEnter(Collision other)
    {
        // 부딛히면 끝장이지.
        if (other.collider.CompareTag("Obstacle"))
        {
            SetReward(-1.0f);
            EndEpisode();      
        }
 
    }


}
