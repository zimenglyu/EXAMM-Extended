#include <cmath>
using std::isfinite;

#include <cstdlib>

#include <memory>
using std::make_shared;
using std::shared_ptr;

#include <string>
using std::string;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/evaluation.hxx"
#include "rnn/rnn.hxx"

string MSELoss::name() const {
    return "mse";
}

LossEvaluation MSELoss::evaluate(RNN& rnn, const vector<vector<double> >& expected_outputs) const {
    return {rnn.calculate_error_mse(expected_outputs)};
}

double MSELoss::optimization_objective(double reported_value) const {
    return 0.5 * reported_value * reported_value;
}

void MSELoss::backward(
    RNN& rnn, const vector<vector<double> >& expected_outputs, const LossEvaluation& evaluation, bool using_dropout,
    bool training, double dropout_probability
) const {
    rnn.backward_pass(
        evaluation.reported_value * (1.0 / expected_outputs[0].size()) * 2.0, using_dropout, training,
        dropout_probability
    );
}

string MinimumValidationMSECheckpointPolicy::name() const {
    return "minimum_validation_mse";
}

bool MinimumValidationMSECheckpointPolicy::should_replace(
    const ValidationMetrics& candidate, const ValidationMetrics& current_best
) const {
    if (!isfinite(candidate.mse)) {
        return false;
    }
    return candidate.mse < current_best.mse;
}

string MinimumValidationMAECheckpointPolicy::name() const {
    return "minimum_validation_mae";
}

bool MinimumValidationMAECheckpointPolicy::should_replace(
    const ValidationMetrics& candidate, const ValidationMetrics& current_best
) const {
    if (!isfinite(candidate.mae)) {
        return false;
    }
    return candidate.mae < current_best.mae;
}

string ValidationMSEFitnessPolicy::name() const {
    return "validation_mse";
}

double ValidationMSEFitnessPolicy::score(const ValidationMetrics& metrics) const {
    return metrics.mse;
}

string ValidationMAEFitnessPolicy::name() const {
    return "validation_mae";
}

double ValidationMAEFitnessPolicy::score(const ValidationMetrics& metrics) const {
    return metrics.mae;
}

EvaluationPlan::EvaluationPlan(
    shared_ptr<const TrainingLoss> training_loss, shared_ptr<const CheckpointPolicy> checkpoint_policy,
    shared_ptr<const FitnessPolicy> fitness_policy
)
    : training_loss_value(training_loss), checkpoint_policy_value(checkpoint_policy), fitness_policy_value(fitness_policy) {}

const TrainingLoss& EvaluationPlan::training_loss() const {
    return *training_loss_value;
}

const CheckpointPolicy& EvaluationPlan::checkpoint_policy() const {
    return *checkpoint_policy_value;
}

const FitnessPolicy& EvaluationPlan::fitness_policy() const {
    return *fitness_policy_value;
}

shared_ptr<const FitnessPolicy> EvaluationPlan::fitness_policy_handle() const {
    return fitness_policy_value;
}

const EvaluationPlan& default_evaluation_plan() {
    static const EvaluationPlan plan(
        make_shared<MSELoss>(), make_shared<MinimumValidationMSECheckpointPolicy>(),
        make_shared<ValidationMSEFitnessPolicy>()
    );
    return plan;
}

shared_ptr<const EvaluationPlan> evaluation_plan_from_arguments(const vector<string>& arguments) {
    string training_loss_name = "mse";
    get_argument(arguments, "--training_loss", false, training_loss_name);
    if (training_loss_name != "mse") {
        Log::fatal(
            "ERROR: unsupported --training_loss '%s'; only 'mse' is currently supported.\n",
            training_loss_name.c_str()
        );
        exit(1);
    }

    string checkpoint_metric = "mse";
    get_argument(arguments, "--checkpoint_metric", false, checkpoint_metric);
    shared_ptr<const CheckpointPolicy> checkpoint_policy;
    if (checkpoint_metric == "mse") {
        checkpoint_policy = make_shared<MinimumValidationMSECheckpointPolicy>();
    } else if (checkpoint_metric == "mae") {
        checkpoint_policy = make_shared<MinimumValidationMAECheckpointPolicy>();
    } else {
        Log::fatal("ERROR: unsupported --checkpoint_metric '%s'; expected 'mse' or 'mae'.\n", checkpoint_metric.c_str());
        exit(1);
    }

    string fitness_metric = "mse";
    get_argument(arguments, "--fitness_metric", false, fitness_metric);
    shared_ptr<const FitnessPolicy> fitness_policy;
    if (fitness_metric == "mse") {
        fitness_policy = make_shared<ValidationMSEFitnessPolicy>();
    } else if (fitness_metric == "mae") {
        fitness_policy = make_shared<ValidationMAEFitnessPolicy>();
    } else {
        Log::fatal("ERROR: unsupported --fitness_metric '%s'; expected 'mse' or 'mae'.\n", fitness_metric.c_str());
        exit(1);
    }

    shared_ptr<const EvaluationPlan> plan = make_shared<EvaluationPlan>(
        make_shared<MSELoss>(), checkpoint_policy, fitness_policy
    );
    Log::info(
        "Evaluation plan: training loss '%s', checkpoint policy '%s', fitness '%s'\n",
        plan->training_loss().name().c_str(), plan->checkpoint_policy().name().c_str(),
        plan->fitness_policy().name().c_str()
    );
    return plan;
}
