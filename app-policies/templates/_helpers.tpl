{{/* validate that action is audit or enforce */}}
{{- define "appPolicies.action" -}}
{{- $v := . -}}
{{- if not (or (eq $v "audit") (eq $v "enforce")) -}}
{{- fail (printf "policy action must be 'audit' or 'enforce', got: %s" $v) -}}
{{- end -}}
{{- $v -}}
{{- end -}}
