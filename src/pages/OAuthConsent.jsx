import React, { useEffect, useState } from "react";
import { appParams } from "@/lib/app-params";
import { Button } from "@/components/ui/button";
import { ShieldCheck, Loader2, CheckCircle2, AlertOctagon } from "lucide-react";
import AuthLayout from "@/components/AuthLayout";

const OPERATION_VERBS = { read: "view", create: "create", update: "edit", delete: "delete" };
const COLLAPSED_DATA_ROWS = 6;

function groupTools(tools) {
  const operationsByEntity = new Map();
  const agents = [];
  const actions = [];
  for (const tool of tools) {
    if (tool.kind === "entity" && tool.entity) {
      if (!operationsByEntity.has(tool.entity)) operationsByEntity.set(tool.entity, []);
      operationsByEntity.get(tool.entity).push(tool.operation);
    } else if (tool.kind === "agent") {
      agents.push(tool);
    } else {
      actions.push(tool);
    }
  }
  const data = [...operationsByEntity].map(([entity, operations]) => ({
    entity,
    operations: Object.keys(OPERATION_VERBS).filter((op) => operations.includes(op)),
  }));
  return { data, agents, actions };
}

const humanize = (name) =>
  name.split(/[_-]+/).filter(Boolean).map((word) => word[0].toUpperCase() + word.slice(1)).join(" ") || name;

function OperationSummary({ operations }) {
  return (
    <>
      {operations.map((op, i) => {
        const verb = OPERATION_VERBS[op];
        const separator = i === 0 ? "" : i === operations.length - 1 ? " and " : ", ";
        return (
          <React.Fragment key={op}>
            {separator}
            <span className={op === "delete" ? "font-bold text-[#FF5252]" : "text-[#D9DEE3]"}>
              {i === 0 ? verb[0].toUpperCase() + verb.slice(1) : verb}
            </span>
          </React.Fragment>
        );
      })}{" "}
      records
    </>
  );
}

function ToolSection({ title, children }) {
  return (
    <section className="space-y-2">
      <h3 className="text-[10px] font-bold uppercase tracking-widest text-[#7A838D]">{title}</h3>
      <ul className="divide-y divide-[#1E232A] rounded-xl border border-[#1E232A] bg-[#0F1216] overflow-hidden">{children}</ul>
    </section>
  );
}

function ToolRow({ title, children }) {
  return (
    <li className="flex flex-col gap-0.5 px-3.5 py-2.5 text-xs">
      <span className="font-semibold text-[#F2F5F7]">{title}</span>
      {children && <span className="text-[#7A838D]">{children}</span>}
    </li>
  );
}

export default function OAuthConsent() {
  const ctx = new URLSearchParams(window.location.search).get("ctx");
  const [info, setInfo] = useState(null);
  const [checking, setChecking] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [decided, setDecided] = useState("");
  const [error, setError] = useState("");
  const [reconnect, setReconnect] = useState("");
  const [showAllData, setShowAllData] = useState(false);

  useEffect(() => {
    (async () => {
      let redirecting = false;
      try {
        if (!ctx) {
          setError("This authorization link is invalid or has expired.");
          return;
        }
        const infoHeaders = {};
        if (appParams.token) infoHeaders.Authorization = "Bearer " + appParams.token;
        const res = await fetch(
          `/api/apps/${appParams.appId}/mcp/consent-info?handle=${encodeURIComponent(ctx)}`,
          { credentials: "include", headers: infoHeaders },
        );
        if (!res.ok) {
          setError("This authorization link is invalid or has expired.");
          return;
        }
        const data = await res.json();
        if (!data.authenticated) {
          const returnTo = window.location.pathname + "?ctx=" + encodeURIComponent(ctx);
          const encoded = encodeURIComponent(returnTo);
          redirecting = true;
          window.location.href =
            (data.login_path || "/login") + "?returnTo=" + encoded + "&from_url=" + encoded;
          return;
        }
        setInfo(data);
      } catch (e) {
        setError("Could not load this authorization request. Please try again.");
      } finally {
        if (!redirecting) setChecking(false);
      }
    })();
  }, [ctx]);

  const respond = async (action) => {
    setSubmitting(true);
    setError("");
    try {
      const headers = { "Content-Type": "application/json" };
      if (appParams.token) headers.Authorization = "Bearer " + appParams.token;
      const res = await fetch(`/api/apps/${appParams.appId}/mcp/authorize-grant`, {
        method: "POST",
        credentials: "include",
        headers,
        body: JSON.stringify({ ctx, action }),
      });
      if (!res.ok) {
        if (res.status === 401) {
          const returnTo = window.location.pathname + "?ctx=" + encodeURIComponent(ctx);
          const encoded = encodeURIComponent(returnTo);
          window.location.href =
            ((info && info.login_path) || "/login") + "?returnTo=" + encoded + "&from_url=" + encoded;
          return;
        }
        if ([400, 403, 404, 409].includes(res.status)) {
          let detail = "";
          try { detail = (await res.json()).detail; } catch (_) {}
          setReconnect(detail || "This authorization can no longer be completed. Reconnect from your AI client to try again.");
          setSubmitting(false);
          return;
        }
        throw new Error("Could not complete authorization. Please try again.");
      }
      const data = await res.json();
      window.location.href = data.redirect_url;
      if (!/^https?:/i.test(data.redirect_url)) {
        setDecided(action);
        setSubmitting(false);
      }
    } catch (e) {
      setError(e.message);
      setSubmitting(false);
    }
  };

  if (checking) {
    return (
      <AuthLayout icon={ShieldCheck} title="Authorize Access">
        <div className="flex flex-col items-center justify-center py-8 text-[#7A838D] space-y-3">
          <Loader2 className="w-6 h-6 text-[#2ED88A] animate-spin" />
          <span className="text-xs font-semibold uppercase tracking-wider">Verifying Link...</span>
        </div>
      </AuthLayout>
    );
  }

  const client = (info && info.client_name) || "An AI client";
  const appName = (info && info.app_name) || "this app";

  if (decided) {
    return (
      <AuthLayout
        icon={decided === "approve" ? CheckCircle2 : AlertOctagon}
        title={decided === "approve" ? "Access Granted" : "Access Denied"}
        subtitle={`You can return to ${client} and close this window.`}
      />
    );
  }

  if (reconnect) {
    return (
      <AuthLayout icon={AlertOctagon} title="Reconnect Required">
        <div className="p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium">
          {reconnect}
        </div>
      </AuthLayout>
    );
  }

  if (error && !info) {
    return (
      <AuthLayout icon={ShieldCheck} title="Authorize Access">
        <div className="p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium">
          {error}
        </div>
      </AuthLayout>
    );
  }

  const tools = Array.isArray(info.tools) ? info.tools : [];
  const { data, agents, actions } = groupTools(tools);

  return (
    <AuthLayout
      icon={ShieldCheck}
      title="Authorize Access"
      subtitle={`${client} wants to access ${appName} on your behalf`}
    >
      {error && (
        <div className="mb-4 p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium">
          {error}
        </div>
      )}

      <p className="text-xs font-bold uppercase tracking-wider text-[#F2F5F7] mb-3">
        {tools.length ? `${client} will be able to:` : "No tools requested"}
      </p>

      {tools.length > 0 && (
        <div className="space-y-4 mb-6">
          {data.length > 0 && (
            <ToolSection title={`Data Permissions (${data.length})`}>
              {(showAllData ? data : data.slice(0, COLLAPSED_DATA_ROWS)).map(({ entity, operations }) => (
                <ToolRow key={entity} title={entity}>
                  <OperationSummary operations={operations} />
                </ToolRow>
              ))}
              {data.length > COLLAPSED_DATA_ROWS && (
                <li className="px-3.5 py-2">
                  <button
                    type="button"
                    className="text-xs font-bold text-[#2ED88A] hover:underline"
                    onClick={() => setShowAllData((shown) => !shown)}
                  >
                    {showAllData ? "Show Less" : `Show All ${data.length}`}
                  </button>
                </li>
              )}
            </ToolSection>
          )}
          {agents.length > 0 && (
            <ToolSection title="Agents">
              {agents.map((tool) => (
                <ToolRow key={tool.name} title={tool.agent ? humanize(tool.agent) : tool.title || tool.name}>
                  {tool.description}
                </ToolRow>
              ))}
            </ToolSection>
          )}
          {actions.length > 0 && (
            <ToolSection title="Actions">
              {actions.map((tool) => (
                <ToolRow key={tool.name} title={tool.title || tool.name}>
                  {tool.description}
                </ToolRow>
              ))}
            </ToolSection>
          )}
        </div>
      )}

      <div className="flex gap-3">
        <Button
          variant="outline"
          className="flex-1 h-12 text-xs font-bold uppercase tracking-wider border-[#1E232A] text-[#7A838D] hover:bg-[#1E232A] hover:text-[#F2F5F7] rounded-xl transition-all"
          disabled={submitting}
          onClick={() => respond("deny")}
        >
          Deny
        </Button>
        <Button
          className="flex-1 h-12 text-xs font-bold uppercase tracking-wider bg-[#2ED88A] text-[#07080A] hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] rounded-xl transition-all"
          disabled={submitting}
          onClick={() => respond("approve")}
        >
          {submitting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
          Approve
        </Button>
      </div>
    </AuthLayout>
  );
}
