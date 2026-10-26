#include <algorithm>
using std::sort;

#include <cmath>
using std::fabs;
using std::isfinite;

#include <cstdlib>
using std::free;

#include <iostream>
using std::cerr;
using std::endl;

#include <memory>
using std::make_shared;
using std::shared_ptr;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/evaluation.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

namespace {

bool nearly_equal(double left, double right, double tolerance = 1e-7) {
    double scale = 1.0 + fabs(left) + fabs(right);
    return fabs(left - right) <= tolerance * scale;
}

bool equal_vectors(const vector<double>& left, const vector<double>& right, double tolerance = 1e-7) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t i = 0; i < left.size(); i++) {
        if (!nearly_equal(left[i], right[i], tolerance)) {
            return false;
        }
    }
    return true;
}

void require(bool condition, const string& message) {
    if (!condition) {
        cerr << "FAIL: " << message << endl;
        exit(EXIT_FAILURE);
    }
}

void require_weighted_shapes(
    const vector<vector<double> >& predictions, const vector<vector<double> >& expected_outputs,
    const vector<vector<double> >& weights
) {
    require(!expected_outputs.empty(), "weighted loss requires at least one output");
    require(predictions.size() == expected_outputs.size(), "weighted loss prediction/output count must match");
    require(weights.size() == expected_outputs.size(), "weighted loss weights/output count must match");

    for (size_t output = 0; output < expected_outputs.size(); output++) {
        require(
            predictions[output].size() == expected_outputs[output].size(),
            "weighted loss prediction/output sequence lengths must match"
        );
        require(
            weights[output].size() == expected_outputs[output].size(),
            "weighted loss coefficient/output sequence lengths must match"
        );
    }
}

/**
 * Test-only loss with a deliberately different derivative at every output and
 * timestep.  It exercises the public vector-output-gradient extension used by
 * future losses; MSELoss intentionally uses EXAMM's scalar legacy BPTT path.
 */
class WeightedSquaredErrorLoss final : public TrainingLoss {
   private:
    vector<vector<double> > weights;

   public:
    explicit WeightedSquaredErrorLoss(vector<vector<double> > _weights) : weights(_weights) {}

    string name() const override {
        return "weighted_squared_error";
    }

    LossEvaluation evaluate(RNN& rnn, const vector<vector<double> >& expected_outputs) const override {
        const vector<vector<double> > predictions = rnn.get_output_values();
        require_weighted_shapes(predictions, expected_outputs, weights);

        const double time_count = static_cast<double>(expected_outputs[0].size());
        double loss = 0.0;
        for (size_t output = 0; output < expected_outputs.size(); output++) {
            for (size_t time = 0; time < expected_outputs[output].size(); time++) {
                const double residual = predictions[output][time] - expected_outputs[output][time];
                loss += 0.5 * weights[output][time] * residual * residual / time_count;
            }
        }
        return {loss};
    }

    double optimization_objective(double reported_value) const override {
        return reported_value;
    }

    void backward(
        RNN& rnn, const vector<vector<double> >& expected_outputs, const LossEvaluation& evaluation,
        bool using_dropout, bool training, double dropout_probability
    ) const override {
        (void) evaluation;
        const vector<vector<double> > predictions = rnn.get_output_values();
        require_weighted_shapes(predictions, expected_outputs, weights);

        const double time_count = static_cast<double>(expected_outputs[0].size());
        vector<vector<double> > output_gradients(expected_outputs.size());
        for (size_t output = 0; output < expected_outputs.size(); output++) {
            output_gradients[output].resize(expected_outputs[output].size());
            for (size_t time = 0; time < expected_outputs[output].size(); time++) {
                output_gradients[output][time] =
                    weights[output][time] * (predictions[output][time] - expected_outputs[output][time]) / time_count;
            }
        }
        rnn.backward_pass(output_gradients, using_dropout, training, dropout_probability);
    }
};

/** Test-only policy which proves that stochastic BPTT consults the supplied checkpoint policy. */
class RecordingCheckpointPolicy final : public CheckpointPolicy {
   private:
    bool replace;

   public:
    mutable int calls = 0;

    explicit RecordingCheckpointPolicy(bool _replace) : replace(_replace) {}

    string name() const override {
        return replace ? "always_replace" : "always_keep";
    }

    bool should_replace(const ValidationMetrics& candidate, const ValidationMetrics& current_best) const override {
        (void) candidate;
        (void) current_best;
        calls++;
        return replace;
    }
};

double independently_calculated_mse(
    RNN& rnn, const vector<double>& parameters, const vector<vector<double> >& inputs,
    const vector<vector<double> >& outputs
) {
    rnn.set_weights(parameters);
    vector<double> predictions = rnn.get_predictions(inputs, outputs, false, 0.0);

    double mse = 0.0;
    for (size_t output = 0; output < outputs.size(); output++) {
        double output_sum = 0.0;
        for (size_t time = 0; time < outputs[output].size(); time++) {
            double prediction = predictions[time * outputs.size() + output];
            double error = prediction - outputs[output][time];
            output_sum += error * error;
        }
        mse += output_sum / outputs[output].size();
    }
    return mse;
}

double mse_objective(
    RNN& rnn, const vector<double>& parameters, const vector<vector<double> >& inputs,
    const vector<vector<double> >& outputs
) {
    double mse = independently_calculated_mse(rnn, parameters, inputs, outputs);
    return 0.5 * mse * mse;
}

vector<double> finite_difference_gradient(
    RNN& rnn, const vector<double>& parameters, const vector<vector<double> >& inputs,
    const vector<vector<double> >& outputs
) {
    const double step = 1e-6;
    vector<double> gradient(parameters.size(), 0.0);
    vector<double> perturbed = parameters;

    for (size_t i = 0; i < perturbed.size(); i++) {
        const double original = perturbed[i];

        perturbed[i] = original - step;
        const double lower = mse_objective(rnn, perturbed, inputs, outputs);

        perturbed[i] = original + step;
        const double upper = mse_objective(rnn, perturbed, inputs, outputs);

        gradient[i] = (upper - lower) / (2.0 * step);
        perturbed[i] = original;
    }
    return gradient;
}

double independently_calculated_weighted_squared_error(
    RNN& rnn, const vector<double>& parameters, const vector<vector<double> >& inputs,
    const vector<vector<double> >& outputs, const vector<vector<double> >& weights
) {
    rnn.set_weights(parameters);
    rnn.forward_pass(inputs, false, true, 0.0);
    const vector<vector<double> > predictions = rnn.get_output_values();
    require_weighted_shapes(predictions, outputs, weights);

    const double time_count = static_cast<double>(outputs[0].size());
    double loss = 0.0;
    for (size_t output = 0; output < outputs.size(); output++) {
        for (size_t time = 0; time < outputs[output].size(); time++) {
            const double residual = predictions[output][time] - outputs[output][time];
            loss += 0.5 * weights[output][time] * residual * residual / time_count;
        }
    }
    return loss;
}

vector<double> finite_difference_weighted_squared_error_gradient(
    RNN& rnn, const vector<double>& parameters, const vector<vector<double> >& inputs,
    const vector<vector<double> >& outputs, const vector<vector<double> >& weights
) {
    const double step = 1e-6;
    vector<double> gradient(parameters.size(), 0.0);
    vector<double> perturbed = parameters;

    for (size_t i = 0; i < perturbed.size(); i++) {
        const double original = perturbed[i];

        perturbed[i] = original - step;
        const double lower = independently_calculated_weighted_squared_error(rnn, perturbed, inputs, outputs, weights);

        perturbed[i] = original + step;
        const double upper = independently_calculated_weighted_squared_error(rnn, perturbed, inputs, outputs, weights);

        gradient[i] = (upper - lower) / (2.0 * step);
        perturbed[i] = original;
    }
    return gradient;
}

size_t recurrent_parameter_start(RNN& rnn) {
    size_t start = 0;
    for (int32_t node = 0; node < rnn.get_number_nodes(); node++) {
        start += static_cast<size_t>(rnn.get_node(node)->get_number_weights());
    }
    return start + static_cast<size_t>(rnn.get_number_edges());
}

shared_ptr<const EvaluationPlan> mae_fitness_plan() {
    return make_shared<EvaluationPlan>(
        make_shared<MSELoss>(), make_shared<MinimumValidationMSECheckpointPolicy>(),
        make_shared<ValidationMAEFitnessPolicy>()
    );
}

void verify_mse_gradient_parity_and_numeric_gradient() {
    WeightRules weight_rules;
    vector<string> input_names{"x0", "x1"};
    vector<string> output_names{"y0", "y1"};
    RNN_Genome* genome = create_ff(input_names, 1, 1, output_names, 1, &weight_rules);

    vector<double> parameters;
    genome->get_weights(parameters);
    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i] = -0.23 + 0.071 * static_cast<double>(i);
    }

    const vector<vector<double> > inputs{{0.25, -0.40, 0.80}, {-0.10, 0.30, 0.55}};
    const vector<vector<double> > outputs{{0.15, -0.20, 0.65}, {-0.35, 0.10, 0.40}};

    RNN* default_rnn = genome->get_rnn();
    RNN* explicit_rnn = genome->get_rnn();
    RNN* empirical_rnn = genome->get_rnn();
    RNN* numerical_rnn = genome->get_rnn();

    double default_mse = 0.0;
    double explicit_mse = 0.0;
    double empirical_mse = 0.0;
    vector<double> default_gradient;
    vector<double> explicit_gradient;
    vector<double> empirical_gradient;
    default_rnn->get_analytic_gradient(
        parameters, inputs, outputs, default_mse, default_gradient, false, true, 0.0
    );
    explicit_rnn->get_analytic_gradient(
        parameters, inputs, outputs, explicit_mse, explicit_gradient, false, true, 0.0,
        default_evaluation_plan().training_loss()
    );
    empirical_rnn->get_empirical_gradient(
        parameters, inputs, outputs, empirical_mse, empirical_gradient, false, true, 0.0,
        default_evaluation_plan().training_loss()
    );

    const double independent_mse = independently_calculated_mse(*numerical_rnn, parameters, inputs, outputs);
    require(nearly_equal(default_mse, independent_mse), "reported MSE must retain its original multi-output value");
    require(nearly_equal(default_mse, explicit_mse), "explicit MSE loss must report MSE");
    require(
        equal_vectors(default_gradient, explicit_gradient, 1e-12),
        "default analytic gradient must match the explicit MSELoss gradient"
    );

    const vector<double> numeric_gradient = finite_difference_gradient(*numerical_rnn, parameters, inputs, outputs);
    require(
        equal_vectors(explicit_gradient, numeric_gradient, 2e-5),
        "analytic gradient must match finite differences of 0.5 * MSE^2"
    );
    require(nearly_equal(empirical_mse, independent_mse), "MSE empirical gradient must report the MSE value");
    require(
        equal_vectors(explicit_gradient, empirical_gradient, 2e-4),
        "analytic MSE BPTT gradient must match the built-in empirical-gradient implementation"
    );

    const size_t recurrent_start = recurrent_parameter_start(*explicit_rnn);
    require(recurrent_start < parameters.size(), "MSE gradient fixture must contain recurrent-edge parameters");
    bool found_nonzero_recurrent_gradient = false;
    for (size_t parameter = recurrent_start; parameter < parameters.size(); parameter++) {
        require(
            nearly_equal(explicit_gradient[parameter], numeric_gradient[parameter], 2e-5),
            "every recurrent-edge MSE gradient must match finite differences"
        );
        found_nonzero_recurrent_gradient = found_nonzero_recurrent_gradient || fabs(explicit_gradient[parameter]) > 1e-8;
    }
    require(found_nonzero_recurrent_gradient, "MSE fixture must exercise a nonzero recurrent-edge gradient");

    delete default_rnn;
    delete explicit_rnn;
    delete empirical_rnn;
    delete numerical_rnn;
    delete genome;
}

void verify_custom_loss_gradient_and_recurrent_coverage() {
    WeightRules weight_rules;
    vector<string> input_names{"x0", "x1"};
    vector<string> output_names{"y0", "y1"};
    RNN_Genome* genome = create_ff(input_names, 1, 1, output_names, 1, &weight_rules);

    vector<double> parameters;
    genome->get_weights(parameters);
    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i] = -0.19 + 0.053 * static_cast<double>(i);
    }

    const vector<vector<double> > inputs{{0.15, -0.30, 0.55, 0.80}, {-0.25, 0.45, 0.10, -0.60}};
    const vector<vector<double> > outputs{{0.20, -0.15, 0.40, 0.75}, {-0.30, 0.05, 0.65, -0.20}};
    const vector<vector<double> > weights{{0.25, 1.50, 0.70, 2.00}, {1.75, 0.40, 2.30, 0.90}};
    WeightedSquaredErrorLoss loss(weights);

    RNN* analytic_rnn = genome->get_rnn();
    RNN* empirical_rnn = genome->get_rnn();
    RNN* numerical_rnn = genome->get_rnn();

    double analytic_loss = 0.0;
    double empirical_loss = 0.0;
    vector<double> analytic_gradient;
    vector<double> empirical_gradient;
    analytic_rnn->get_analytic_gradient(
        parameters, inputs, outputs, analytic_loss, analytic_gradient, false, true, 0.0, loss
    );
    empirical_rnn->get_empirical_gradient(
        parameters, inputs, outputs, empirical_loss, empirical_gradient, false, true, 0.0, loss
    );

    const double independent_loss =
        independently_calculated_weighted_squared_error(*numerical_rnn, parameters, inputs, outputs, weights);
    const vector<double> numeric_gradient =
        finite_difference_weighted_squared_error_gradient(*numerical_rnn, parameters, inputs, outputs, weights);

    require(
        nearly_equal(analytic_loss, independent_loss),
        "custom training loss must report its independently calculated objective"
    );
    require(
        nearly_equal(empirical_loss, independent_loss),
        "custom empirical gradient must report its independently calculated objective"
    );
    require(
        equal_vectors(analytic_gradient, numeric_gradient, 3e-5),
        "custom-loss analytic BPTT gradient must match independent finite differences"
    );
    require(
        equal_vectors(empirical_gradient, numeric_gradient, 3e-4),
        "custom-loss empirical gradient must match independent finite differences"
    );
    require(
        equal_vectors(analytic_gradient, empirical_gradient, 3e-4),
        "custom-loss analytic and empirical gradients must agree"
    );

    const size_t recurrent_start = recurrent_parameter_start(*analytic_rnn);
    require(recurrent_start < parameters.size(), "custom-loss fixture must contain recurrent-edge parameters");
    bool found_nonzero_recurrent_gradient = false;
    for (size_t parameter = recurrent_start; parameter < parameters.size(); parameter++) {
        require(
            nearly_equal(analytic_gradient[parameter], numeric_gradient[parameter], 3e-5),
            "custom-loss recurrent-edge gradient must match independent finite differences"
        );
        found_nonzero_recurrent_gradient = found_nonzero_recurrent_gradient || fabs(analytic_gradient[parameter]) > 1e-8;
    }
    require(found_nonzero_recurrent_gradient, "custom loss fixture must exercise a nonzero recurrent-edge gradient");

    delete analytic_rnn;
    delete empirical_rnn;
    delete numerical_rnn;
    delete genome;
}

void verify_checkpoint_and_fitness_policies() {
    MinimumValidationMSECheckpointPolicy mse_checkpoint;
    MinimumValidationMAECheckpointPolicy mae_checkpoint;
    ValidationMSEFitnessPolicy mse_fitness;
    ValidationMAEFitnessPolicy mae_fitness;

    const ValidationMetrics best{1.0, 0.5};
    require(mse_checkpoint.should_replace({0.9, 5.0}, best), "lower validation MSE must replace a checkpoint");
    require(!mse_checkpoint.should_replace({1.0, 0.1}, best), "equal validation MSE must keep the first checkpoint");
    require(!mse_checkpoint.should_replace({NAN, 0.1}, best), "NaN validation MSE must not replace a checkpoint");
    require(
        !mse_checkpoint.should_replace({0.1, 0.1}, {NAN, NAN}),
        "a legacy NaN checkpoint must retain its original non-replacement behavior"
    );
    require(mae_checkpoint.should_replace({5.0, 0.4}, best), "lower validation MAE must replace an MAE checkpoint");
    require(!mae_checkpoint.should_replace({0.0, 0.5}, best), "equal validation MAE must keep the first checkpoint");
    require(nearly_equal(mse_fitness.score(best), 1.0), "MSE fitness must use validation MSE");
    require(nearly_equal(mae_fitness.score(best), 0.5), "MAE fitness must use validation MAE");
}

void verify_evaluation_plan_argument_parsing() {
    const ValidationMetrics current_best{1.0, 0.5};
    const ValidationMetrics metric_pair{0.8, 0.2};

    const EvaluationPlan& default_plan = default_evaluation_plan();
    require(default_plan.training_loss().name() == "mse", "default training loss must be MSE");
    require(
        default_plan.checkpoint_policy().name() == "minimum_validation_mse",
        "default checkpoint policy must select validation MSE"
    );
    require(default_plan.fitness_policy().name() == "validation_mse", "default fitness policy must use validation MSE");

    vector<string> no_evaluation_arguments{"test"};
    const shared_ptr<const EvaluationPlan> parsed_default = evaluation_plan_from_arguments(no_evaluation_arguments);
    require(parsed_default->training_loss().name() == "mse", "omitted training-loss option must parse as MSE");
    require(
        parsed_default->checkpoint_policy().name() == "minimum_validation_mse",
        "omitted checkpoint option must parse as validation MSE"
    );
    require(
        parsed_default->fitness_policy().name() == "validation_mse",
        "omitted fitness option must parse as validation MSE"
    );

    const vector<string> metric_names{"mse", "mae"};
    for (const string& checkpoint_metric : metric_names) {
        for (const string& fitness_metric : metric_names) {
            const vector<string> arguments{
                "test", "--training_loss", "mse", "--checkpoint_metric", checkpoint_metric, "--fitness_metric",
                fitness_metric
            };
            const shared_ptr<const EvaluationPlan> plan = evaluation_plan_from_arguments(arguments);

            require(plan->training_loss().name() == "mse", "explicit MSE training loss must parse as MSE");
            require(
                plan->checkpoint_policy().name()
                    == (checkpoint_metric == "mse" ? "minimum_validation_mse" : "minimum_validation_mae"),
                "checkpoint metric option must choose the corresponding checkpoint policy"
            );
            require(
                plan->fitness_policy().name()
                    == (fitness_metric == "mse" ? "validation_mse" : "validation_mae"),
                "fitness metric option must choose the corresponding fitness policy"
            );

            const ValidationMetrics candidate =
                checkpoint_metric == "mse" ? ValidationMetrics{0.9, 0.9} : ValidationMetrics{1.1, 0.4};
            require(
                plan->checkpoint_policy().should_replace(candidate, current_best),
                "parsed checkpoint policy must compare the selected validation metric"
            );
            const double expected_fitness = fitness_metric == "mse" ? metric_pair.mse : metric_pair.mae;
            require(
                nearly_equal(plan->fitness_policy().score(metric_pair), expected_fitness),
                "parsed fitness policy must score the selected validation metric"
            );
        }
    }
}

void verify_policy_attachment_and_mpi_wire_round_trip() {
    WeightRules weight_rules;
    vector<string> input_names{"x"};
    vector<string> output_names{"y"};
    RNN_Genome* genome = create_ff(input_names, 0, 0, output_names, 0, &weight_rules);

    vector<double> parameters;
    genome->get_weights(parameters);
    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i] = 0.17 * static_cast<double>(i + 1);
    }
    genome->set_initial_parameters(parameters);
    genome->set_best_parameters(parameters);
    genome->set_validation_metrics({0.8, 0.3});

    shared_ptr<const EvaluationPlan> plan = mae_fitness_plan();
    genome->set_evaluation_plan(*plan);
    require(nearly_equal(genome->get_fitness(), 0.3), "attached MAE fitness policy must drive scalar fitness");

    char* bytes = nullptr;
    int32_t length = 0;
    genome->write_to_array(&bytes, length);
    RNN_Genome* received = new RNN_Genome(bytes, length);
    free(bytes);

    require(
        nearly_equal(received->get_best_validation_mse(), 0.8)
            && nearly_equal(received->get_best_validation_mae(), 0.3),
        "validation metrics must survive the exact MPI array serializer"
    );
    require(
        nearly_equal(received->get_fitness(), 0.8),
        "an unconfigured deserialized genome must safely retain default MSE fitness"
    );
    received->set_evaluation_plan(*plan);
    require(
        nearly_equal(received->get_fitness(), 0.3),
        "rank-local plan reattachment must restore the selected fitness policy"
    );

    delete received;
    delete genome;
}

void verify_fitness_policy_survives_evolutionary_copy() {
    WeightRules weight_rules;
    vector<string> input_names{"x"};
    vector<string> output_names{"y"};
    RNN_Genome* mse_best_mae_worse = create_ff(input_names, 0, 0, output_names, 0, &weight_rules);
    RNN_Genome* mse_worse_mae_best = create_ff(input_names, 0, 0, output_names, 0, &weight_rules);
    const shared_ptr<const EvaluationPlan> plan = mae_fitness_plan();

    mse_best_mae_worse->set_validation_metrics({0.1, 1.0});
    mse_worse_mae_best->set_validation_metrics({0.2, 0.1});
    mse_best_mae_worse->set_evaluation_plan(*plan);
    mse_worse_mae_best->set_evaluation_plan(*plan);

    RNN_Genome* copied_mse_best_mae_worse = mse_best_mae_worse->copy();
    RNN_Genome* copied_mse_worse_mae_best = mse_worse_mae_best->copy();
    vector<RNN_Genome*> population{copied_mse_best_mae_worse, copied_mse_worse_mae_best};
    sort(population.begin(), population.end(), sort_genomes_by_fitness());

    require(
        population.front() == copied_mse_worse_mae_best,
        "evolutionary copies must retain the selected MAE fitness policy when genomes are sorted"
    );
    require(
        nearly_equal(copied_mse_worse_mae_best->get_fitness(), 0.1),
        "copied genome fitness must use its retained MAE policy"
    );

    delete copied_mse_best_mae_worse;
    delete copied_mse_worse_mae_best;
    delete mse_best_mae_worse;
    delete mse_worse_mae_best;
}

RNN_Genome* make_recurrent_multi_output_genome(WeightRules& weight_rules) {
    vector<string> input_names{"x0", "x1"};
    vector<string> output_names{"y0", "y1"};
    return create_ff(input_names, 1, 1, output_names, 1, &weight_rules);
}

void configure_one_step_backprop(RNN_Genome& genome, const vector<double>& initial_parameters) {
    genome.set_initial_parameters(initial_parameters);
    genome.set_bp_iterations(1);
    genome.disable_dropout();
    genome.set_weight_decay(0.0);
}

WeightUpdate make_vanilla_weight_update(double learning_rate) {
    const vector<string> arguments{"test", "--weight_update", "vanilla"};
    WeightUpdate update(arguments);
    update.set_learning_rate(learning_rate);
    update.disable_high_threshold();
    update.disable_low_threshold();
    return update;
}

void verify_stochastic_custom_loss_and_checkpoint_routing() {
    WeightRules weight_rules;
    RNN_Genome* replacing_genome = make_recurrent_multi_output_genome(weight_rules);
    RNN_Genome* keeping_genome = make_recurrent_multi_output_genome(weight_rules);

    vector<double> initial_parameters;
    replacing_genome->get_weights(initial_parameters);
    for (size_t i = 0; i < initial_parameters.size(); i++) {
        initial_parameters[i] = -0.16 + 0.041 * static_cast<double>(i);
    }
    configure_one_step_backprop(*replacing_genome, initial_parameters);
    configure_one_step_backprop(*keeping_genome, initial_parameters);

    const vector<vector<vector<double> > > inputs{{{0.15, -0.30, 0.55, 0.80}, {-0.25, 0.45, 0.10, -0.60}}};
    const vector<vector<vector<double> > > outputs{{{0.20, -0.15, 0.40, 0.75}, {-0.30, 0.05, 0.65, -0.20}}};
    const vector<vector<double> > weights{{0.25, 1.50, 0.70, 2.00}, {1.75, 0.40, 2.30, 0.90}};
    const shared_ptr<const TrainingLoss> loss = make_shared<WeightedSquaredErrorLoss>(weights);

    RNN* gradient_rnn = replacing_genome->get_rnn();
    double reported_loss = 0.0;
    vector<double> analytic_gradient;
    gradient_rnn->get_analytic_gradient(
        initial_parameters, inputs[0], outputs[0], reported_loss, analytic_gradient, false, true, 0.0, *loss
    );
    const double independently_calculated_loss =
        independently_calculated_weighted_squared_error(*gradient_rnn, initial_parameters, inputs[0], outputs[0], weights);
    require(
        nearly_equal(reported_loss, independently_calculated_loss),
        "the stochastic-update reference gradient must use the custom loss"
    );

    const double learning_rate = 0.001;
    vector<double> expected_updated_parameters = initial_parameters;
    for (size_t i = 0; i < expected_updated_parameters.size(); i++) {
        expected_updated_parameters[i] -= learning_rate * analytic_gradient[i];
    }
    const double updated_loss = independently_calculated_weighted_squared_error(
        *gradient_rnn, expected_updated_parameters, inputs[0], outputs[0], weights
    );
    require(
        updated_loss < independently_calculated_loss,
        "one deterministic vanilla update must reduce the custom training objective"
    );
    delete gradient_rnn;

    const ValidationMetrics initial_metrics{
        replacing_genome->get_mse(initial_parameters, inputs, outputs),
        replacing_genome->get_mae(initial_parameters, inputs, outputs)
    };
    const ValidationMetrics updated_metrics{
        replacing_genome->get_mse(expected_updated_parameters, inputs, outputs),
        replacing_genome->get_mae(expected_updated_parameters, inputs, outputs)
    };

    const shared_ptr<RecordingCheckpointPolicy> always_replace = make_shared<RecordingCheckpointPolicy>(true);
    const EvaluationPlan replace_plan(loss, always_replace, make_shared<ValidationMAEFitnessPolicy>());
    WeightUpdate replace_update = make_vanilla_weight_update(learning_rate);
    replacing_genome->backpropagate_stochastic(inputs, outputs, inputs, outputs, &replace_update, replace_plan);

    require(always_replace->calls == 1, "stochastic BPTT must call the supplied replacement policy once per epoch");
    require(
        equal_vectors(replacing_genome->get_best_parameters(), expected_updated_parameters, 1e-12),
        "stochastic BPTT must apply the custom-loss gradient with the expected sign and scale"
    );
    require(
        nearly_equal(replacing_genome->get_best_validation_mse(), updated_metrics.mse)
            && nearly_equal(replacing_genome->get_best_validation_mae(), updated_metrics.mae),
        "an accepting checkpoint policy must retain the updated MSE and MAE together"
    );
    require(
        nearly_equal(replacing_genome->get_fitness(), updated_metrics.mae),
        "custom checkpoint selection must remain independent from MAE population fitness"
    );

    const shared_ptr<RecordingCheckpointPolicy> always_keep = make_shared<RecordingCheckpointPolicy>(false);
    const EvaluationPlan keep_plan(loss, always_keep, make_shared<ValidationMAEFitnessPolicy>());
    WeightUpdate keep_update = make_vanilla_weight_update(learning_rate);
    keeping_genome->backpropagate_stochastic(inputs, outputs, inputs, outputs, &keep_update, keep_plan);

    require(always_keep->calls == 1, "stochastic BPTT must call the supplied keep policy once per epoch");
    require(
        equal_vectors(keeping_genome->get_best_parameters(), initial_parameters, 1e-12),
        "a rejecting checkpoint policy must retain the initial checkpoint parameters"
    );
    require(
        nearly_equal(keeping_genome->get_best_validation_mse(), initial_metrics.mse)
            && nearly_equal(keeping_genome->get_best_validation_mae(), initial_metrics.mae),
        "a rejecting checkpoint policy must retain the initial MSE and MAE together"
    );
    require(
        nearly_equal(keeping_genome->get_fitness(), initial_metrics.mae),
        "a rejecting checkpoint policy must still expose MAE population fitness"
    );

    delete replacing_genome;
    delete keeping_genome;
}

void verify_default_stochastic_plan_parity() {
    WeightRules weight_rules;
    RNN_Genome* default_genome = make_recurrent_multi_output_genome(weight_rules);
    RNN_Genome* explicit_genome = make_recurrent_multi_output_genome(weight_rules);

    vector<double> initial_parameters;
    default_genome->get_weights(initial_parameters);
    for (size_t i = 0; i < initial_parameters.size(); i++) {
        initial_parameters[i] = -0.12 + 0.037 * static_cast<double>(i);
    }
    configure_one_step_backprop(*default_genome, initial_parameters);
    configure_one_step_backprop(*explicit_genome, initial_parameters);

    const vector<vector<vector<double> > > inputs{{{0.10, -0.20, 0.35}, {0.25, 0.15, -0.40}}};
    const vector<vector<vector<double> > > outputs{{{0.05, 0.30, -0.25}, {-0.10, 0.45, 0.20}}};
    WeightUpdate default_update = make_vanilla_weight_update(0.001);
    WeightUpdate explicit_update = make_vanilla_weight_update(0.001);

    default_genome->backpropagate_stochastic(inputs, outputs, inputs, outputs, &default_update);
    explicit_genome->backpropagate_stochastic(
        inputs, outputs, inputs, outputs, &explicit_update, default_evaluation_plan()
    );

    require(
        equal_vectors(default_genome->get_best_parameters(), explicit_genome->get_best_parameters(), 1e-12),
        "default stochastic BPTT overload must match an explicit default evaluation plan"
    );
    require(
        nearly_equal(default_genome->get_best_validation_mse(), explicit_genome->get_best_validation_mse())
            && nearly_equal(default_genome->get_best_validation_mae(), explicit_genome->get_best_validation_mae()),
        "default stochastic BPTT overload must retain the same validation metrics as an explicit default plan"
    );
    require(
        nearly_equal(default_genome->get_fitness(), explicit_genome->get_fitness()),
        "default stochastic BPTT overload must attach the same fitness policy as an explicit default plan"
    );

    delete default_genome;
    delete explicit_genome;
}

void verify_stochastic_checkpoint_integration() {
    WeightRules weight_rules;
    vector<string> input_names{"x"};
    vector<string> output_names{"y"};
    RNN_Genome* genome = create_ff(input_names, 0, 0, output_names, 0, &weight_rules);

    vector<double> initial_parameters;
    genome->get_weights(initial_parameters);
    for (size_t i = 0; i < initial_parameters.size(); i++) {
        initial_parameters[i] = -0.11 + 0.09 * static_cast<double>(i);
    }
    genome->set_initial_parameters(initial_parameters);
    genome->set_bp_iterations(3);

    const vector<vector<vector<double> > > training_inputs{{{0.10, -0.30, 0.70}}};
    const vector<vector<vector<double> > > training_outputs{{{0.20, -0.15, 0.60}}};
    const vector<vector<vector<double> > > validation_inputs{{{0.05, 0.25, -0.40}}};
    const vector<vector<vector<double> > > validation_outputs{{{0.15, 0.30, -0.10}}};

    const double expected_mse = genome->get_mse(initial_parameters, validation_inputs, validation_outputs);
    const double expected_mae = genome->get_mae(initial_parameters, validation_inputs, validation_outputs);

    WeightUpdate update;
    update.set_learning_rate(0.0);
    update.disable_high_threshold();
    update.disable_low_threshold();

    shared_ptr<const EvaluationPlan> plan = mae_fitness_plan();
    genome->backpropagate_stochastic(
        training_inputs, training_outputs, validation_inputs, validation_outputs, &update, *plan
    );

    require(
        equal_vectors(genome->get_best_parameters(), initial_parameters, 1e-12),
        "zero-learning-rate ties must retain the initial checkpoint parameters"
    );
    require(
        nearly_equal(genome->get_best_validation_mse(), expected_mse)
            && nearly_equal(genome->get_best_validation_mae(), expected_mae),
        "the retained checkpoint must report its MSE and MAE together"
    );
    require(nearly_equal(genome->get_fitness(), expected_mae), "fitness must remain independent from checkpoint metric");
    require(isfinite(genome->get_fitness()), "trained genome must have finite scalar fitness");

    delete genome;
}

void add_default_log_argument(vector<string>& arguments, const string& option, const string& value) {
    if (!argument_exists(arguments, option)) {
        arguments.push_back(option);
        arguments.push_back(value);
    }
}

}  // namespace

int main(int argc, char** argv) {
    vector<string> arguments(argv, argv + argc);
    add_default_log_argument(arguments, "--std_message_level", "NONE");
    add_default_log_argument(arguments, "--file_message_level", "NONE");
    add_default_log_argument(arguments, "--output_directory", "test_loss_checkpoint_fitness_logs");
    Log::initialize(arguments);
    Log::set_id("test_loss_checkpoint_fitness");

    verify_mse_gradient_parity_and_numeric_gradient();
    verify_custom_loss_gradient_and_recurrent_coverage();
    verify_checkpoint_and_fitness_policies();
    verify_evaluation_plan_argument_parsing();
    verify_policy_attachment_and_mpi_wire_round_trip();
    verify_fitness_policy_survives_evolutionary_copy();
    verify_stochastic_custom_loss_and_checkpoint_routing();
    verify_default_stochastic_plan_parity();
    verify_stochastic_checkpoint_integration();

    cerr << "PASS: loss, evaluation-plan, checkpoint, fitness, and BPTT gradient checks" << endl;
    return EXIT_SUCCESS;
}
