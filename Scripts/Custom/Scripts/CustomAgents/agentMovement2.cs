using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class AgentMovement : Agent
{
    // Target
    public Transform Goal;
    public Transform MyPos;
    float goalDistance;
    float preDistance;

    // Vessel Physcial parameter
    public float maxPower = 0.05f;
    public float maxRudderAngle = 0.05f;
    public float maxSpeed = 2f; // 최대 속도를 정의
    
    // 추가로 필요한 Vessel Physical parameter
    public float dragCof;

    // RayCast & LineRenderer
    public float rayLength;     
    public LineRenderer lineRenderer;    
    public int rayCount;           
    public float raySpreadAngle;  
    public float[] rayObjDis;            

    // Unity Basic Component (Essential)
    private Rigidbody rd;

    public override void Initialize() {

        // Get Basic Component 
        rd = GetComponent<Rigidbody>();

        // Hyperparmeter Vessel
        maxPower = 0.05f; // 최대 전진 속도
        maxRudderAngle = 0.3f; // 최대 러더 각도

        // Hyperparmeter RayCast 
        rayLength = 10.0f;
        rayCount = 360;
        raySpreadAngle = 1f;

        MyPos = this.gameObject.transform;
    }

    public override void OnEpisodeBegin()
    {
        //My Position

        this.gameObject.transform.position = MyPos.position;


        goalDistance = Vector3.Distance(Goal.position, this.gameObject.transform.position);
        preDistance = goalDistance;

        // RayCast & LineRenderer 매 Episode마다 호출
        rayObjDis = new float[rayCount];   // 각 레이의 거리 정보 배열 초기화
        for (int i = 0; i < rayCount; i++) rayObjDis[i] = rayLength;  // 초기 거리를 최대 레이 길이로 설정

        lineRenderer = gameObject.AddComponent<LineRenderer>();      // 라인 렌더러 컴포넌트 추가
        lineRenderer.positionCount = rayCount * 2;  // 레이의 시작점과 끝점
        lineRenderer.startWidth = 0.01f;  // 라인의 시작 폭
        lineRenderer.endWidth = 0.01f;    // 라인의 끝 폭
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));  // 라인의 재질
        lineRenderer.startColor = Color.red;  // 라인의 시작 색
        lineRenderer.endColor = Color.red;    // 라인의 끝 색

    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float powerAction = actions.ContinuousActions[0];  // Power
        float rudderAction = actions.ContinuousActions[1]; // Rudder

        agentMovement(powerAction, rudderAction);


        // Find fastest way
        AddReward(-0.0001f);
        // Reward Setting
        if (goalDistance < 1.0f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (goalDistance > 10.0f)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
        else
        {
            // Reward 세팅
            preDistance = Vector3.Distance(Goal.position, this.gameObject.transform.position);

            if(preDistance < goalDistance)
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

        if (rd.velocity.magnitude > maxSpeed)
        {
            rd.velocity = rd.velocity.normalized * maxSpeed;
        }

        float power = maxPower * Mathf.Clamp(powerAction, -1f, 1f); // 전진 속도 조정
        float rudderAngle = maxRudderAngle * Mathf.Clamp(rudderAction, -1f, 1f); // 러더 각도 조정

        // Turn
        Quaternion turnRotation = Quaternion.Euler(0f, rudderAngle, 0f);
        rd.MoveRotation(rd.rotation * turnRotation);

        // Go - X축으로 전진 변경
        rd.AddForce(transform.right * power, ForceMode.VelocityChange);
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * raySpreadAngle - 180;  // 각도를 계산하여 360도 전체에 레이를 균등하게 분포
            Quaternion rotation = Quaternion.Euler(0f, angle, 0f);  // 회전값 설정
            Vector3 direction = rotation * transform.forward;  // 방향 벡터 계산

            Ray ray = new Ray(transform.position, direction);  // 레이 생성
            RaycastHit hit;  // 레이캐스트 히트 정보

            int index = i * 2;  // 라인 렌더러의 위치 인덱스 계산
            lineRenderer.SetPosition(index, ray.origin);  // 라인의 시작점 설정

            if (Physics.Raycast(ray, out hit, rayLength))
            {
                // 레이가 객체에 충돌한 경우
                lineRenderer.SetPosition(index + 1, hit.point);  // 라인의 끝점을 충돌 지점으로 설정
                rayObjDis[i] = hit.distance;  // 충돌 거리 저장
            }
            else
            {
                // 레이가 충돌하지 않은 경우
                lineRenderer.SetPosition(index + 1, ray.origin + ray.direction * rayLength);  // 라인의 끝점을 최대 길이로 설정
                rayObjDis[i] = rayLength;  // 최대 거리로 설정
            }
        }

        // RayCast 360개 Obs
        sensor.AddObservation(rayObjDis);

        // My Position - 2개 <Message Actor용>
        sensor.AddObservation(this.gameObject.transform.position);

        // Distance from Goal - 1개
        sensor.AddObservation(Vector3.Distance(Goal.position, this.gameObject.transform.position));
       
        

    }

   


}
