%% Kinetic_modelling.m
% GA + fmincon approach with single overall R^2, ensuring no parameter can be zero.
% 17 parameters: (14 kinetic + 3 initial conditions).
% Minimizes SSR/SST => maximizing R^2 across X, S, P.

clear; clc; close all;

%% SECTION 1: Load Experimental Data
filename = 'wee_data.csv';
expData = readtable(filename);

t_exp = expData.Time;
X_exp = expData.Biomass;
S_exp = expData.Sugar;
P_exp = expData.LacticAcid;

% Sort by time just in case
[~, idxSort] = sort(t_exp);
t_exp = t_exp(idxSort);
X_exp = X_exp(idxSort);
S_exp = S_exp(idxSort);
P_exp = P_exp(idxSort);

t_start = t_exp(1);
t_final = t_exp(end);

%% SECTION 2: Parameter Setup
numParams = 17;

% Lower bounds => all > 1e-6 (no zero)
lb_all = ones(numParams,1)*1e-6;

% Upper bounds for the 14 kinetic parameters
ub_kin = [
    3
    5
    400
    50
    0.1
    10
    5
    400
    50
    1
    10
    5
    400
    100
];
% For the 3 initial conditions
ub_IC = [2; 200; 10];
ub_all = [ub_kin; ub_IC];

%% GA options
optionsGA = optimoptions('ga',...
    'PopulationSize', 100,...
    'MaxGenerations', 300,...
    'Display','iter',...
    'UseParallel', true,...   % parallel if you have the Parallel Toolbox
    'HybridFcn', @fmincon);   % auto local refinement

%% Objective Function
fitnessFcn = @(p) objectiveFunction_R2total_noZero(p, t_exp, X_exp, S_exp, P_exp);

fprintf('\n=== GA + HybridFcn = fmincon (Overall R^2) ===\n');
[bestParams, bestFval] = ga(fitnessFcn, numParams, [], [], [], [], lb_all, ub_all, [], optionsGA);

fprintf('\n--- Final GA + HybridFcn Results ---\n');
disp('Best Parameter Set (14 kinetics + X0, S0, P0):');
disp(bestParams);
fprintf('Objective Function Value (SSR/SST): %f\n', bestFval);

%% Solve ODE and Compute Final R^2
X0_best = bestParams(15);
S0_best = bestParams(16);
P0_best = bestParams(17);

tspan = [t_start, t_final];
y0 = [X0_best; S0_best; P0_best];

[t_sim, y_sim] = ode15s(@(tt,xx) lactic_acid_model(tt,xx,bestParams), tspan, y0);

X_sim = y_sim(:,1);
S_sim = y_sim(:,2);
P_sim = y_sim(:,3);

X_plucked = interp1(t_sim, X_sim, t_exp, 'linear','extrap');
S_plucked = interp1(t_sim, S_sim, t_exp, 'linear','extrap');
P_plucked = interp1(t_sim, P_sim, t_exp, 'linear','extrap');

modelAll = [X_plucked; S_plucked; P_plucked];
expAll   = [X_exp;     S_exp;     P_exp];

SSR = sum((modelAll - expAll).^2);
meanExp = mean(expAll);
SST = sum((expAll - meanExp).^2);

Q_final = SSR / SST;  % Minimizing => maximizing total R^2
R2_total = 1 - Q_final;

fprintf('\nFinal SSR/SST = %f\n', Q_final);
fprintf('Final Overall R^2 = %f\n', R2_total);

%% Display Comparison Table
comparisonTable = table(t_exp, X_exp, X_plucked, S_exp, S_plucked, P_exp, P_plucked,...
    'VariableNames',{'Time','X_Exp','X_Sim','S_Exp','S_Sim','P_Exp','P_Sim'});
disp('--- Experimental vs. Final Simulation ---');
disp(comparisonTable);

%% Plot
figure('Name','GA + fmincon Best Fit (No Zero Param, Single R^2, Parallel+Hybrid)');
subplot(3,1,1);
plot(t_sim, X_sim, 'r-', 'LineWidth',1.5); hold on;
plot(t_exp, X_exp, 'ro', 'MarkerFaceColor','r');
xlabel('Time (h)'); ylabel('Biomass (g/L)');
title(sprintf('Biomass (X), R^2_{total}=%.3f', R2_total));
legend('Sim','Exp','Location','Best');
grid on;

subplot(3,1,2);
plot(t_sim, S_sim, 'b-', 'LineWidth',1.5); hold on;
plot(t_exp, S_exp, 'bo', 'MarkerFaceColor','b');
xlabel('Time (h)'); ylabel('Sugar (g/L)');
title(sprintf('Sugar (S), R^2_{total}=%.3f', R2_total));
legend('Sim','Exp','Location','Best');
grid on;

subplot(3,1,3);
plot(t_sim, P_sim, 'g-', 'LineWidth',1.5); hold on;
plot(t_exp, P_exp, 'go', 'MarkerFaceColor','g');
xlabel('Time (h)'); ylabel('Lactic Acid (g/L)');
title(sprintf('Lactic Acid (P), R^2_{total}=%.3f', R2_total));
legend('Sim','Exp','Location','Best');
grid on;

%% Parameter Table
paramNames = {
    'mumax','Ksx','Kix','Kpx','Kd','qsmax','Kss','Kis','Kps',...
    'alpha','qpmax','Ksp','Kip','Kpp','X0','S0','P0'
};
paramValues = bestParams(:);
paramTable = table(paramNames', paramValues,...
    'VariableNames', {'Parameter','Value'});
disp('--- Final Parameter Values ---');
disp(paramTable);

%% LOCAL FUNCTION DEFINITIONS MUST GO LAST

function dxdt = lactic_acid_model(~, x, p)
    % p(1..14) = kinetic, p(15..17) = X0, S0, P0 (unused in ODE).
    X = x(1);
    S = x(2);
    P = x(3);

    mumax = p(1);
    Ksx   = p(2);
    Kix   = p(3);
    Kpx   = p(4);
    Kd    = p(5);
    qsmax = p(6);
    Kss   = p(7);
    Kis   = p(8);
    Kps   = p(9);
    alpha = p(10);
    qpmax = p(11);
    Ksp   = p(12);
    Kip   = p(13);
    Kpp   = p(14);

    mu = (mumax * S * Kix) / ((Ksx + S)*(Kix + S)) * exp(-P / Kpx);

    dXdt = (mu - Kd)*X;
    dSdt = -qsmax*(S*Kis)/((Kss + S)*(Kis + S)) * exp(-P / Kps)*X;
    dPdt = alpha*dXdt + qpmax*(S*Kip)/((Ksp + S)*(Kip + S)) * exp(-P / Kpp)*X;

    dxdt = [dXdt; dSdt; dPdt];
end

function Q = objectiveFunction_R2total_noZero(p, t_exp, X_exp, S_exp, P_exp)
    % Single overall R^2 approach => Q = SSR/SST across X,S,P (min => max R^2).
    % Ensures all param > 1e-6 => no zeros.

    % If ANY parameter < 1e-6, penalize:
    if any(p < 1e-6)
        Q = Inf;
        return;
    end

    X0 = p(15);
    S0 = p(16);
    P0 = p(17);

    t_start = t_exp(1);
    t_final = t_exp(end);
    y0 = [X0; S0; P0];

    try
        [t_sim, y_sim] = ode15s(@(tt,xx) lactic_acid_model(tt,xx,p), [t_start, t_final], y0);

        X_model = interp1(t_sim, y_sim(:,1), t_exp, 'linear','extrap');
        S_model = interp1(t_sim, y_sim(:,2), t_exp, 'linear','extrap');
        P_model = interp1(t_sim, y_sim(:,3), t_exp, 'linear','extrap');

        modelAll = [X_model; S_model; P_model];
        expAll   = [X_exp;   S_exp;   P_exp];

        SSR = sum((modelAll - expAll).^2);
        meanExp = mean(expAll);
        SST = sum((expAll - meanExp).^2);

        Q = SSR / SST;  % Minimizing => maximizing total R^2 = 1 - Q

    catch
        Q = Inf;
    end
end
