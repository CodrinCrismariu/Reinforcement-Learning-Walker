using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

using BodyPart = Unity.MLAgentsExamples.BodyPart;
using Random = UnityEngine.Random;

public class WalkerAgent : Agent
{
    [Header("Walk Speed")]
    [Range(0.1f, 10)]
    [SerializeField]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = 10;

    public float MTargetWalkingSpeed // property
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    public BehaviorParameters agentBehaviorParams;

    const float m_maxWalkingSpeed = 10; //The max walking speed

    public Unity.InferenceEngine.ModelAsset recoveryModel;
    public Unity.InferenceEngine.ModelAsset walkingModel;

    // Call this method to switch to Model A
    public void UseRecoveryModel()
    {
        // "YourBehaviorName" should match the name in the BehaviorParameters
        SetModel("RecoveryBehavior", recoveryModel);
    }

    // Call this method to switch to Model B
    public void UseWalkingModel()
    {
        SetModel("WalkingBehavior", walkingModel);
    }

    //Should the agent sample a new goal velocity each episode?
    //If true, walkSpeed will be randomly set between zero and m_maxWalkingSpeed in OnEpisodeBegin()
    //If false, the goal velocity will be walkingSpeed
    public bool randomizeWalkSpeedEachEpisode;

    public enum LearningStage
    {
        Easy = 0,
        Medium = 1,
        Hard = 2,
        Walking = 3
    }

    [Header("Initialization")]
    public LearningStage startingStage = LearningStage.Walking;
    public bool terminateOnBodyPartContact = false;

    [Tooltip("Maximum penalty per step for knees touching ground (scales with time)")]
    public float maxKneeTouchPenalty = -5f;

    [Tooltip("Ratio of gravity to compensate (0.0 = no assist, 0.5 = half weight)")]
    public float currAssistForceRatio = 1f; // Can be controlled by Curriculum

    [Header("Gait Rewards")]
    public float oneFootReward = 0.05f;
    public float doubleSupportPenalty = -0.05f;

    float m_TotalMass;

    //The direction an agent will walk during training.
    private Vector3 m_WorldDirToWalk = Vector3.right;

    [Header("Target To Walk Towards")] public Transform target; //Target the agent will walk towards during training.

    [Header("Body Parts")] public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;
    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        agentBehaviorParams = GetComponent<BehaviorParameters>();

        //Setup each body part
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(chest);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(head);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(footL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);
        m_JdController.SetupBodyPart(footR);
        m_JdController.SetupBodyPart(armL);
        m_JdController.SetupBodyPart(forearmL);
        m_JdController.SetupBodyPart(handL);
        m_JdController.SetupBodyPart(armR);
        m_JdController.SetupBodyPart(forearmR);
        m_JdController.SetupBodyPart(handR);

        // Calculate Total Mass
        m_TotalMass = 0f;
        foreach (var bp in m_JdController.bodyPartsList)
        {
            m_TotalMass += bp.rb.mass;
        }

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        //Reset all of the body parts
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        hips.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        // Read difficulty from environment parameters (Project Settings -> ML-Agents -> Environment Parameters)
        // This allows overriding the stage via curriculum or CLI arguments.
        // 0 = Easy, 1 = Medium, 2 = Hard, 3 = Walking
        float difficulty = m_ResetParams.GetWithDefault("difficulty", (float)startingStage);
        startingStage = (LearningStage)Mathf.RoundToInt(difficulty);

        // Read Assist Force from Curriculum
        // Param "assist_force" corresponds to the ratio of weight to support (0 to 1)
        currAssistForceRatio = m_ResetParams.GetWithDefault("assist_force", currAssistForceRatio);

        // Read generic curriculum value (-1 to 1)
        float curriculumValue = m_ResetParams.GetWithDefault("curriculum_value", -1.0f);
        // You can map this 'curriculumValue' to other parameters if needed

        // Update Ground Contact Termination
        // If we are learning to stand, we usually want to disable termination on body contact.
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            if (bodyPart.rb.transform == footL || bodyPart.rb.transform == footR)
            {
                // Feet should never terminate episode on ground contact
                bodyPart.groundContact.agentDoneOnGroundContact = false;
            }
            else
            {
                bodyPart.groundContact.agentDoneOnGroundContact = terminateOnBodyPartContact;
            }
        }

        // Apply specific stage initialization
        switch (startingStage)
        {
            case LearningStage.Easy:
                // Stage 1 (Easy): Start slightly off-balance but upright.
                // Apply small random rotation on X/Z axes
                hips.rotation *= Quaternion.Euler(Random.Range(-10f, 10f), 0, Random.Range(-10f, 10f));
                break;
            case LearningStage.Medium:
                // Stage 2 (Medium): Start on knees.
                // 1. Lower hips to put knees near ground (assuming ~0.5 height).
                hips.position += new Vector3(0, -0.3f, 0);

                // 2. Bend Knees: Rotate shins back significantly so agent is kneeling.
                // Setting local rotation ensures it's relative to the thigh.
                // -120 degrees usually folds the leg back effectively.
                shinL.localRotation = Quaternion.Euler(120, 0, 0);
                shinR.localRotation = Quaternion.Euler(120, 0, 0);
                thighL.localRotation = Quaternion.Euler(Random.Range(-90f, 90f), 0, 0);
                thighR.localRotation = Quaternion.Euler(Random.Range(-90f, 90f), 0, 0);

                // 3. Tilt hips forward slightly to balance on knees
                hips.rotation *= Quaternion.Euler(-15f, 0, 0);
                break;
            case LearningStage.Hard:
                // Stage 3 (Hard): Start lying flat on the back or stomach (Ragdoll).
                bool faceDown = Random.value > 0.5f;
                // Rotate 90 degrees on X (face down) or -90 (face up)
                float xRot = faceDown ? 90f : -90f;
                hips.rotation = Quaternion.Euler(xRot, Random.Range(0.0f, 360.0f), 0);
                // Lower hips to near ground level
                hips.position = new Vector3(hips.position.x, 0.5f, hips.position.z);
                break;
            case LearningStage.Walking:
            default:
                // Stage 4: Walking (Standard Start)
                // Already handled by default resets above
                break;
        }

        UpdateOrientationObjects();

        //Set our goal walking speed
        MTargetWalkingSpeed =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.1f, m_maxWalkingSpeed) : MTargetWalkingSpeed;
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.linearVelocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));

        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));

        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR)
        {
            sensor.AddObservation(bp.rb.transform.localRotation);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity we want to match
        var velGoal = cubeForward * MTargetWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));

        //rotation deltas
        sensor.AddObservation(Quaternion.FromToRotation(hips.forward, cubeForward));
        sensor.AddObservation(Quaternion.FromToRotation(head.forward, cubeForward));

        //Position of target position relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(target.transform.position));

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        var bpDict = m_JdController.bodyPartsDict;
        var i = -1;

        var continuousActions = actionBuffers.ContinuousActions;
        bpDict[chest].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[spine].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[thighL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[thighR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[shinL].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[footR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[footL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[armL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[armR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[forearmL].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[forearmR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[head].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        //update joint strength settings
        bpDict[chest].SetJointStrength(continuousActions[++i]);
        bpDict[spine].SetJointStrength(continuousActions[++i]);
        bpDict[head].SetJointStrength(continuousActions[++i]);
        bpDict[thighL].SetJointStrength(continuousActions[++i]);
        bpDict[shinL].SetJointStrength(continuousActions[++i]);
        bpDict[footL].SetJointStrength(continuousActions[++i]);
        bpDict[thighR].SetJointStrength(continuousActions[++i]);
        bpDict[shinR].SetJointStrength(continuousActions[++i]);
        bpDict[footR].SetJointStrength(continuousActions[++i]);
        bpDict[armL].SetJointStrength(continuousActions[++i]);
        bpDict[forearmL].SetJointStrength(continuousActions[++i]);
        bpDict[armR].SetJointStrength(continuousActions[++i]);
        bpDict[forearmR].SetJointStrength(continuousActions[++i]);
    }

    //Update OrientationCube and DirectionIndicator
    void UpdateOrientationObjects()
    {
        m_WorldDirToWalk = target.position - hips.position;
        m_OrientationCube.UpdateOrientation(hips, target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    void FixedUpdate()
    {
        
        UpdateOrientationObjects();

        // Calculate angle of hips relative to world up
        float angle = Vector3.Angle(hips.up, Vector3.up);

        // Threshold for switching behaviors (e.g., 40 degrees)
        if (angle > 60f)
        {
            UseRecoveryModel();
        }
        else if(angle < 20f)
        {
            UseWalkingModel();
        }

        if (startingStage == LearningStage.Walking)
        {
            var cubeForward = m_OrientationCube.transform.forward;

            // Set reward for this step according to mixture of the following elements.
            // a. Match target speed
            //This reward will approach 1 if it matches perfectly and approach zero as it deviates
            var matchSpeedReward = GetMatchingVelocityReward(cubeForward * MTargetWalkingSpeed, GetAvgVelocity());

            //Check for NaNs
            if (float.IsNaN(matchSpeedReward))
            {
                throw new ArgumentException(
                    "NaN in moveTowardsTargetReward.\n" +
                    $" cubeForward: {cubeForward}\n" +
                    $" hips.velocity: {m_JdController.bodyPartsDict[hips].rb.linearVelocity}\n" +
                    $" maximumWalkingSpeed: {m_maxWalkingSpeed}"
                );
            }

            // b. Rotation alignment with target direction.
            //This reward will approach 1 if it faces the target direction perfectly and approach zero as it deviates
            var headForward = head.forward;
            headForward.y = 0;
            // var lookAtTargetReward = (Vector3.Dot(cubeForward, head.forward) + 1) * .5F;
            var lookAtTargetReward = (Vector3.Dot(cubeForward, headForward) + 1) * .5F;

            //Check for NaNs
            if (float.IsNaN(lookAtTargetReward))
            {
                throw new ArgumentException(
                    "NaN in lookAtTargetReward.\n" +
                    $" cubeForward: {cubeForward}\n" +
                    $" head.forward: {head.forward}"
                );
            }

            AddReward(matchSpeedReward * lookAtTargetReward);
        }
        else
        {
            // Stages 1-3: Learning to Stand Up
            // Reward needs to be strong enough to encourage lifting the body.

            // 1. Head Height
            // Reward high head position. Assume ~2m is good max height.
            // This encourages legs to straighten.
            float headHeight = head.position.y;
            float heightReward = Mathf.Pow(Mathf.Clamp01(headHeight / 1.7f), 3);

            // 2. Upright Alignment
            // Use Head and Hips.
            float hipsUp = Vector3.Dot(hips.up, Vector3.up);
            float headUp = Vector3.Dot(head.up, Vector3.up);
            
            // Only reward if alignment is positive (upwards). 
            // Lying flat (0) or upside down (<0) should yield 0 reward for this component.
            float alignmentReward = Mathf.Max(0, hipsUp) * 0.2f + Mathf.Max(0, headUp) * 0.2f;

            // Combine
            float totalReward = (heightReward + alignmentReward) * 0.5f;

            AddReward(totalReward);

            // Knee Touch Penalty (Scales with Time)
            // Check if shins are touching the ground
            bool kneesTouching = false;
            foreach (var bp in m_JdController.bodyPartsList)
            {
                if ((bp.rb.transform == thighL || bp.rb.transform == thighR) && bp.groundContact.touchingGround)
                {
                    kneesTouching = true;
                    break;
                }
            }

            if (kneesTouching)
            {
                // Penalty scales from 0 to maxKneeTouchPenalty based on episode progress
                // prevention of staying on knees
                float timeRatio = (float)StepCount / MaxStep;
                AddReward(maxKneeTouchPenalty * timeRatio);
            }
        }

        // Apply Assistive External Force (HoST Paper)
        // Helps the agent stand up by pulling it up by the hips
        if (currAssistForceRatio > 0)
        {
            // F = m * g * ratio
            float upwardForce = m_TotalMass * Physics.gravity.magnitude * currAssistForceRatio;
            // Apply to Hips (Center of Mass approximation)
            var hipsRb = m_JdController.bodyPartsDict[hips].rb;
            hipsRb.AddForce(Vector3.up * upwardForce, ForceMode.Force);
        }
    }

    //Returns the average velocity of all of the body parts
    //Using the velocity of the hips only has shown to result in more erratic movement from the limbs, so...
    //...using the average helps prevent this erratic movement
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.linearVelocity;
        }

        var avgVel = velSum / numOfRb;
        return avgVel;
    }

    //normalized value of the difference in avg speed vs goal walking speed.
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, MTargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / MTargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        if (startingStage == LearningStage.Walking)
        {
            AddReward(1f);
        }
    }
}
