using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class oneAgent : Agent
{
    Rigidbody rb;

    public Transform StartPosition;
    public Transform Goal;

    public float preDist;
    public float distance;

    public float moveSpeed = 15f;
    public float rotationSpeed = 0.1f;

    public float DecisionWaitingTime = 5f;
    float m_currentTime = 0f;

    public override void OnEpisodeBegin()
    {
        // 초기 위치로 재설정
        this.transform.position = StartPosition.position;
        // 에이전트의 속도와 회전 상태 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        preDist = Vector3.Distance(this.transform.position, Goal.position);
    }

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        Academy.Instance.AgentPreStep += WaitTimeInference;
        preDist = Vector3.Distance(this.transform.position, Goal.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        distance = Vector3.Distance(this.transform.position, Goal.position);
        sensor.AddObservation(distance);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        AddReward(-0.01f);

        float horizontalInput = actionBuffers.ContinuousActions[0];
        float verticalInput = actionBuffers.ContinuousActions[1];

        vesselMovement(horizontalInput, verticalInput);

        if (distance <= 0.1f)
        {
            AddReward(1f);
            EndEpisode();
        }
        else if (distance > 50f)
        {
            AddReward(-1f);
            EndEpisode();
        }
        else
        {
            float reward = preDist - distance;
            if (reward > 0f)
            {
                AddReward(reward / 100f);
                preDist = distance;
            }
        }
    }

    public void WaitTimeInference(int action)
    {
        // ML-Agents의 업데이트에 따라 IsCommunicatorOn의 필요성 검토
        if (Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if (m_currentTime >= DecisionWaitingTime)
            {
                m_currentTime = 0f;
                RequestDecision();
            }
            else
            {
                m_currentTime += Time.fixedDeltaTime;
            }
        }
    }

    public void vesselMovement(float hAxis, float vAxis)
    {
        Vector3 inputDir = new Vector3(hAxis, 0, vAxis).normalized;
        rb.velocity = inputDir * moveSpeed;
        transform.LookAt(transform.position + inputDir);
    }
}
