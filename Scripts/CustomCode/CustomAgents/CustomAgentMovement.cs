using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CustomAgentMovement : MonoBehaviour
{

    //RayCast & LineRenderer
    public float rayLength = 5.0f;
    public LineRenderer lineRenderer;
    public int rayCount = 36;
    public float raySpreadAngle = 10f;
    public float[] rayObjDis;

    // 목적지
    public Transform myWayPoint;
    public float distance;

     // 에이전트 오브젝트 관리
   [SerializeField]
    Rigidbody rb;
    public bool isAlive;

    // 배 움직임
    public float moveSpeed = 0.1f;
    public float rotationSpeed = 0.1f;
    

    void Start()
    {
        // 거리 계산
        distance = Vector3.Magnitude(this.gameObject.transform.position - myWayPoint.position);

        // Rigidbody 초기화
        rb = GetComponent<Rigidbody>();
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // RayCast & LineRenderer 컴포넌트 가져오기
        rayObjDis = new float[36];
        for (int i = 0; i < 36; i++) rayObjDis[i] = rayLength;

        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.positionCount = rayCount * 2; // 시작점과 끝점이 각각 필요하므로 *2
        lineRenderer.startWidth = 0.01f;
        lineRenderer.endWidth = 0.01f;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startColor = Color.red;
        lineRenderer.endColor = Color.red;

        // 상태변수
        isAlive = true;
    }

    void Update()
    {
        // 여기 테스트로 GetKeyDown (w,a,d로만 해보기, 실제 DDPG에서는 a,d 즉 Rudder만 넣을거임)
        // 그리고 Agent Drag coefficient 계산 넣어야함

        //Ray
        performRayCast();
        //Movement
        vesselMovement();
    }

    //override void OnEpisodeBegin()
    //{

    //}

    public void vesselMovement()
    {
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");

        //Vector3 moveDirection = transform.forward * verticalInput;
        //Vector3 rotation = new Vector3(0, horizontalInput, 0) * rotationSpeed * Time.deltaTime;

        //rb.AddForce(moveDirection * moveSpeed);
        //rb.AddTorque(rotation);

        Vector3 moveDirection = transform.forward * verticalInput;
        Vector3 velocity = moveDirection * moveSpeed;

        // 현재 속도와 목표 속도 사이를 보간하여 부드러운 이동 적용
        rb.velocity = Vector3.Lerp(rb.velocity, velocity, 0.1f);

        // 부드러운 회전 적용
        float turn = horizontalInput * rotationSpeed * Time.deltaTime;
        Quaternion turnRotation = Quaternion.Euler(0f, turn, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);

    }

    public void performRayCast()
    {
        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * raySpreadAngle - (raySpreadAngle * (rayCount - 1) / 2);
            Quaternion rotation = Quaternion.Euler(0f, angle, 0f);
            Vector3 direction = rotation * transform.forward;

            Ray ray = new Ray(transform.position, direction);
            RaycastHit hit;

            int index = i * 2; // 레이당 시작점과 끝점 2개를 렌더링하기 위해 *2
            lineRenderer.SetPosition(index, ray.origin);

            if (Physics.Raycast(ray, out hit, rayLength))
            {
                // 레이가 충돌한 경우
                lineRenderer.SetPosition(index + 1, hit.point);
                rayObjDis[i] = hit.distance;

            }
            else
            {
                // 레이가 충돌하지 않은 경우
                lineRenderer.SetPosition(index + 1, ray.origin + ray.direction * rayLength);
                rayObjDis[i] = rayLength;

            }
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Obstacle")) // Vessel Agent이든, 땅이든간에 전부 tag Obstacle 처리
        {
            isAlive = false;

        }

    }


}
