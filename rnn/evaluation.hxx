#ifndef EXAMM_EVALUATION_HXX
#define EXAMM_EVALUATION_HXX

#include <memory>
using std::shared_ptr;

#include <string>
using std::string;

#include <vector>
using std::vector;

class RNN;

/**
 * The value reported for a training loss.  Some historical losses optimize a
 * transformation of this value; optimization_objective() makes that explicit
 * so numerical gradient checks can verify the same objective as BPTT.
 */
struct LossEvaluation {
    double reported_value;
};

/** Metrics measured on one validation checkpoint. */
struct ValidationMetrics {
    double mse;
    double mae;
};

/**
 * A training loss prepares the output-node error values and starts BPTT.
 * Custom implementations can inspect RNN::get_output_values() and send their
 * own derivatives through RNN::backward_pass(output_gradients, ...).
 */
class TrainingLoss {
   public:
    virtual ~TrainingLoss() = default;

    virtual string name() const = 0;

    virtual LossEvaluation evaluate(RNN& rnn, const vector<vector<double> >& expected_outputs) const = 0;

    virtual double optimization_objective(double reported_value) const = 0;

    virtual void backward(
        RNN& rnn, const vector<vector<double> >& expected_outputs, const LossEvaluation& evaluation, bool using_dropout,
        bool training, double dropout_probability
    ) const = 0;
};

/**
 * The mean-squared-error loss used by EXAMM.
 *
 * It reports MSE and preserves EXAMM's established BPTT scaling. Consequently
 * its optimization objective is one half of the square of the reported MSE,
 * rather than textbook MSE itself.
 */
class MSELoss final : public TrainingLoss {
   public:
    string name() const override;

    LossEvaluation evaluate(RNN& rnn, const vector<vector<double> >& expected_outputs) const override;

    double optimization_objective(double reported_value) const override;

    void backward(
        RNN& rnn, const vector<vector<double> >& expected_outputs, const LossEvaluation& evaluation, bool using_dropout,
        bool training, double dropout_probability
    ) const override;
};

/** Chooses whether a validation checkpoint replaces the current best one. */
class CheckpointPolicy {
   public:
    virtual ~CheckpointPolicy() = default;

    virtual string name() const = 0;
    virtual bool should_replace(const ValidationMetrics& candidate, const ValidationMetrics& current_best) const = 0;
};

/** Preserves the historical strict lower-validation-MSE checkpoint rule. */
class MinimumValidationMSECheckpointPolicy final : public CheckpointPolicy {
   public:
    string name() const override;
    bool should_replace(const ValidationMetrics& candidate, const ValidationMetrics& current_best) const override;
};

/** Selects checkpoints using strictly lower validation MAE. */
class MinimumValidationMAECheckpointPolicy final : public CheckpointPolicy {
   public:
    string name() const override;
    bool should_replace(const ValidationMetrics& candidate, const ValidationMetrics& current_best) const override;
};

/** Maps checkpoint metrics to the scalar score used by the current evolution code. */
class FitnessPolicy {
   public:
    virtual ~FitnessPolicy() = default;

    virtual string name() const = 0;
    virtual double score(const ValidationMetrics& metrics) const = 0;
};

/** Preserves the historical scalar validation-MSE fitness. */
class ValidationMSEFitnessPolicy final : public FitnessPolicy {
   public:
    string name() const override;
    double score(const ValidationMetrics& metrics) const override;
};

/** Uses validation MAE as the scalar fitness score. */
class ValidationMAEFitnessPolicy final : public FitnessPolicy {
   public:
    string name() const override;
    double score(const ValidationMetrics& metrics) const override;
};

/**
 * Immutable run-level objective configuration.
 *
 * Loss, checkpoint choice, and population fitness are intentionally separate:
 * changing one does not silently change either of the other two.  The current
 * evolutionary implementation still consumes one scalar fitness; a future
 * multi-objective selector can replace that boundary without changing BPTT or
 * checkpointing.
 */
class EvaluationPlan {
   private:
    shared_ptr<const TrainingLoss> training_loss_value;
    shared_ptr<const CheckpointPolicy> checkpoint_policy_value;
    shared_ptr<const FitnessPolicy> fitness_policy_value;

   public:
    EvaluationPlan(
        shared_ptr<const TrainingLoss> training_loss, shared_ptr<const CheckpointPolicy> checkpoint_policy,
        shared_ptr<const FitnessPolicy> fitness_policy
    );

    const TrainingLoss& training_loss() const;
    const CheckpointPolicy& checkpoint_policy() const;
    const FitnessPolicy& fitness_policy() const;
    shared_ptr<const FitnessPolicy> fitness_policy_handle() const;
};

/** The default plan, retaining EXAMM's current behavior exactly. */
const EvaluationPlan& default_evaluation_plan();

/**
 * Builds a run-level plan from optional MPI arguments:
 *   --training_loss mse
 *   --checkpoint_metric mse|mae
 *   --fitness_metric mse|mae
 */
shared_ptr<const EvaluationPlan> evaluation_plan_from_arguments(const vector<string>& arguments);

#endif
